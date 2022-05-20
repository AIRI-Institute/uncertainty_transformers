import numpy as np
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
import torch.nn.functional as F

from utils.utils_sngp import SNGPTrainer

from transformers import BertModel, BertPreTrainedModel
from transformers import (
    ElectraForSequenceClassification,
    BertForSequenceClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers import Trainer

from torch.autograd import Variable

from transformers.trainer_pt_utils import (
    nested_detach,
)
from transformers.file_utils import (
    is_sagemaker_mp_enabled,
)

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

def entropy(x):
    return torch.sum(-x * torch.log(torch.clamp(x, 1e-8, 1)), axis=-1)

def conf(preds, probs, labels):
    conf_scores = torch.where(preds == labels, torch.max(probs, axis=-1).values, 1 - torch.max(probs, axis=-1).values)
    return conf_scores
    
def RAU_loss(probs, labels, unc_threshold=0.5, eps=1e-6):
    preds = torch.argmax(probs, axis=-1)
    conf_scores = conf(preds, probs, labels)
    uncertainty = entropy(probs)
    n_C = conf_scores * (1 - torch.tan(uncertainty))
    n_U = conf_scores * (torch.tan(uncertainty))
    
    n_AC = torch.where((preds == labels) & (uncertainty <= unc_threshold), n_C, torch.tensor(0.).to(labels.device)).sum()
    n_AU = torch.where((preds == labels) & (uncertainty > unc_threshold), n_U, torch.tensor(0.).to(labels.device)).sum()
    n_IC = torch.where((preds != labels) & (uncertainty <= unc_threshold), n_C, torch.tensor(0.).to(labels.device)).sum()
    n_IU = torch.where((preds != labels) & (uncertainty > unc_threshold), n_U, torch.tensor(0.).to(labels.device)).sum()
    loss = torch.log(1 + n_AU / (n_AC + n_AU + eps) + n_IC / (n_IC + n_IU + eps))
    return loss 

def multiclass_metric_loss_fast(represent, target, margin=10, class_num=2, start_idx=1,
                                per_class_norm=False):
    target_list = target.data.tolist()
    dim = represent.data.shape[1]
    indices = []
    for class_idx in range(start_idx, class_num + start_idx):
        indice_i = [i for i, x in enumerate(target_list) if x == class_idx]
        indices.append(indice_i)

    loss_intra = torch.FloatTensor([0]).to(represent.device)
    num_intra = 0
    loss_inter = torch.FloatTensor([0]).to(represent.device)
    num_inter = 0
    for i in range(class_num):
        curr_repr = represent[indices[i]]
        s_k = len(indices[i])
        triangle_matrix = torch.triu(
            (curr_repr.unsqueeze(1) - curr_repr).norm(2, dim=-1)
        )
        buf_loss = torch.sum(1 / dim * (triangle_matrix ** 2))
        if per_class_norm:
            loss_intra += buf_loss / np.max([(s_k ** 2 - s_k), 1]) * 2
        else:
            loss_intra += buf_loss
            num_intra += (curr_repr.shape[0] ** 2 - curr_repr.shape[0]) / 2
        for j in range(i + 1, class_num):
            repr_j = represent[indices[j]]
            s_q = len(indices[j])
            matrix = (curr_repr.unsqueeze(1) - repr_j).norm(2, dim=-1)
            inter_buf_loss = torch.sum(torch.clamp(margin - 1 / dim * (matrix ** 2), min=0))
            if per_class_norm:
                loss_inter += inter_buf_loss / np.max([(s_k * s_q), 1])
            else:
                loss_inter += inter_buf_loss
                num_inter += repr_j.shape[0] * curr_repr.shape[0]
    if num_intra > 0 and not(per_class_norm):
        loss_intra = loss_intra / num_intra
    if num_inter > 0 and not(per_class_norm):
        loss_inter = loss_inter / num_inter
    return loss_intra, loss_inter


def compute_loss_cer(logits, labels, loss, lamb, unpad=False):
    """Computes regularization term for loss with CER
    """
    # here correctness is always 0 or 1
    if unpad:
        # NER case
        logits = logits[torch.nonzero(labels != -100, as_tuple=True)]
        labels = labels[torch.nonzero(labels != -100, as_tuple=True)]
    # suppose that -1 will works for ner and cls
    confidence, prediction = torch.softmax(logits, dim=-1).max(dim=-1)
    correctness = prediction == labels
    correct_confidence = torch.masked_select(confidence, correctness)
    wrong_confidence = torch.masked_select(confidence, ~correctness)
    regularizer = 0
    if unpad:
        # speed up for NER
        regularizer = torch.sum(
            torch.clamp(wrong_confidence.unsqueeze(1) - correct_confidence, min=0)
            ** 2
        )
    else:
        for cc in correct_confidence:
            for wc in wrong_confidence:
                regularizer += torch.clamp(wc - cc, min=0) ** 2
    loss += lamb * regularizer
    return loss


def compute_loss_metric(hiddens, labels, loss, num_labels,
                        margin, lamb_intra, lamb, unpad=False):
    """Computes regularization term for loss with Metric loss
    """
    if unpad:
        hiddens = hiddens[torch.nonzero(labels != -100, as_tuple=True)]
        labels = labels[torch.nonzero(labels != -100, as_tuple=True)]
    class_num = num_labels
    start_idx = 0 if class_num == 2 else 1
    # TODO: define represent, target and margin
    # Get only sentence representaions
    loss_intra, loss_inter = multiclass_metric_loss_fast(
        hiddens,
        labels,
        margin=margin,
        class_num=class_num,
        start_idx=start_idx,
    )
    loss_metric = lamb_intra * loss_intra[0] + lamb * loss_inter[0]
    loss += loss_metric
    return loss


class SelectiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = getattr(kwargs["args"], "task", "cls")
        self.reg_type = getattr(kwargs["args"], "reg_type", "reg-curr")
        self.lamb = getattr(kwargs["args"], "lamb", 0.01)
        self.margin = getattr(kwargs["args"], "margin", 0.5)
        self.lamb_intra = getattr(kwargs["args"], "lamb_intra", 0.5)
        self.unc_threshold = getattr(kwargs["args"], "unc_threshold", 0.5)
        if self.task == "cls":
            self.unpad = False
        else:
            self.unpad = True

    def post_init(self, reg_type, lamb, margin, lamb_intra, unc_threshold):
        """Add regularization params"""
        self.reg_type = reg_type
        self.lamb = lamb
        self.margin = margin
        self.lamb_intra = lamb_intra
        self.unc_threshold = unc_threshold

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        output_hidden_states = True if self.reg_type == "metric" else False
        outputs = model(**inputs, output_hidden_states=output_hidden_states)
        logits = outputs.logits if self.task == "cls" else outputs[0]
        if self.reg_type == "metric":
            hiddens = outputs.hidden_states[-1][:, 0, :] if self.task == "cls" else outputs[1][-1]
            if self.task == "cls":
                del outputs
                torch.cuda.empty_cache()
                outputs = logits
        if model.config.num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        if self.reg_type == "raw":
            pass
        elif self.reg_type == "reg-curr":
            loss = compute_loss_cer(logits, labels, loss, self.lamb,
                                    unpad=self.unpad)
        elif self.reg_type == "metric":
            loss = compute_loss_metric(hiddens, labels, loss,
                                       model.config.num_labels,
                                       self.margin, self.lamb_intra, self.lamb,
                                       unpad=self.unpad)
            if self.task == "ner":
                # we don't need hiddens anymore
                outputs = outputs[0]
        elif self.reg_type == "rau":
            loss += self.lamb * RAU_loss(torch.softmax(logits, dim=1), labels, self.unc_threshold)
        else:
            raise NotImplementedError()
        if isinstance(outputs, tuple):
            return (loss,) + outputs if return_outputs else loss
        else:
            return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    loss, outputs = self.compute_loss(
                        model, inputs, return_outputs=True
                    )
                    loss = loss.mean().detach()
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        # Changed from original code - there was outputs[1:] for some reason
                        logits = outputs
                else:
                    loss = None
                    if self.use_amp:
                        with autocast():
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


class SelectiveSNGPTrainer(SelectiveTrainer, SNGPTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_type = getattr(kwargs["args"], "reg_type", "reg-curr")
        self.lamb = getattr(kwargs["args"], "lamb", 0.01)
        self.margin = getattr(kwargs["args"], "margin", 0.5)
        self.lamb_intra = getattr(kwargs["args"], "lamb_intra", 0.5)
        self.unc_threshold = getattr(kwargs["args"], "unc_threshold", 0.5)
