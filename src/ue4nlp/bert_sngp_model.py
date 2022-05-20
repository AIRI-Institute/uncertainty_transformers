from typing import Optional, Tuple
from dataclasses import dataclass
import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    ElectraForTokenClassification
)
from transformers.activations import get_activation
from transformers.file_utils import ModelOutput
from ue4nlp.transformers_cached import CachedInferenceMixin
from utils.spectral_norm import spectral_norm
from torch.nn.utils import spectral_norm as pt_spectral_norm
from sklearn.metrics import accuracy_score, auc, roc_auc_score
import numpy as np
import math


@dataclass
class SequenceClassifierOutputSNGP(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        cov_matrix (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels, config.num_labels)`):
            cov_matrix
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    cov_matrix: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GPOutputLayer(torch.nn.Module):
    def __init__(self, config, ue_config):
        super().__init__()

        self.gp_hidden_dim = ue_config.gp_hidden_dim
        self.num_labels = config.num_labels

        self.momentum = ue_config.momentum
        self.ridge_factor = ue_config.ridge_factor
        self.use_layer_norm = ue_config.use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(config.hidden_size)

        self.use_paper_version = ue_config.get("use_paper_version", False)
        if self.use_paper_version:
            self.linear = torch.nn.utils.spectral_norm(
                torch.nn.Linear(config.hidden_size, self.gp_hidden_dim)
            )
        else:
            self.linear = torch.nn.Linear(config.hidden_size, self.gp_hidden_dim)
        # init weight and bias
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
        torch.nn.init.uniform_(self.linear.bias, a=0.0, b=2.0 * math.pi)
        # freeze parameters
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        
        self.precision_matrix = torch.nn.parameter.Parameter(self.ridge_factor * torch.eye(self.gp_hidden_dim).repeat(1, 1), requires_grad=False)
        self.output = torch.nn.Linear(self.gp_hidden_dim, self.num_labels, bias=False)
        if not self.use_paper_version:
            self.output_bias = torch.nn.parameter.Parameter(torch.tensor([0.]*self.num_labels), requires_grad=False)

    def forward(self, h, is_final_epoch, is_first_minibatch, is_last_minibatch, epoch):
        if self.use_layer_norm:
            h = self.layer_norm(h)

        if self.use_paper_version:
            Phi = 2 * math.sqrt(2 / self.gp_hidden_dim) * torch.cos(-self.linear(h))
            output = self.output(Phi)
        else:
            gp_input_scale = 1 / math.sqrt(2)
            Phi = torch.cos(self.linear(gp_input_scale * h))
            #Phi = 2 * math.sqrt(2 / self.gp_hidden_dim) * torch.cos(-self.linear(h))

            output = self.output(Phi) + self.output_bias

        if self.training:
            is_first_minibatch = (epoch == 0) & (is_first_minibatch == True)
            self.update_precision_matrix(
                Phi, output, is_first_minibatch, is_last_minibatch
            )
            gp_cov_matrix = self.compute_predictive_covariance(Phi)
        else:
            gp_cov_matrix = self.compute_predictive_covariance(Phi)

        return output, gp_cov_matrix

    def update_precision_matrix(
        self, Phi, output, is_first_minibatch, is_last_minibatch
    ):
        with torch.no_grad():

            if self.use_paper_version:
                probas = torch.nn.functional.softmax(output, dim=0)
            else:
                probas = output#torch.nn.functional.softmax(output, dim=0)
        
            prob_multiplier = torch.ones(1).to(output.device)
            # prob_multiplier = probas[:, i:i+1] * (1 - probas[:, i:i+1])

            gp_feature_adjusted = torch.sqrt(prob_multiplier) * Phi
            
            precision_matrix_minibatch = gp_feature_adjusted.T @ gp_feature_adjusted
            if not self.use_paper_version:
                precision_matrix_minibatch = precision_matrix_minibatch / Phi.shape[0]

            if is_first_minibatch:
                self.precision_matrix.data = self.ridge_factor * torch.eye(
                    Phi.shape[-1]
                ).repeat(1, 1).to(Phi.device)

            self.precision_matrix.data = (
                self.momentum * self.precision_matrix.data
                + (1 - self.momentum) * precision_matrix_minibatch
            )

            # compute covariance matrix
            if is_last_minibatch:
                self.cov_matrix = torch.inverse(self.precision_matrix)

    def compute_predictive_covariance(self, gp_feature):
        # Computes the covariance matrix of the gp prediction.
        self.cov_matrix = torch.inverse(self.precision_matrix)
        cov_feature_product = (self.cov_matrix @ gp_feature.T) * self.ridge_factor
        gp_cov_matrix = gp_feature @ cov_feature_product

        return gp_cov_matrix


class SpectralNormalizedPooler(torch.nn.Module):
    def __init__(self, pooler, sn_value=0.95, use_paper_version=False):
        super().__init__()
        if use_paper_version:
            self.dense = pt_spectral_norm(
                pooler.dense
            )
        else:
            self.dense = spectral_norm(
                pooler.dense, sn_value
            )
        self.activation = pooler.activation

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0, :]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SNGPBertForSequenceClassificationCached(
    CachedInferenceMixin, BertForSequenceClassification
):
    def __init__(self, config, ue_config):
        super().__init__(config)
        self.classifier = GPOutputLayer(config, ue_config)
        self.bert.pooler = SpectralNormalizedPooler(self.bert.pooler)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_final_epoch=False,
        is_first_minibatch=False,
        is_last_minibatch=False,
        epoch=0,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.inference_body(
            self.bert,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits, cov_matrix = self.classifier(
            pooled_output, is_final_epoch, is_first_minibatch, is_last_minibatch, epoch
        )

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (
                logits,
                cov_matrix,
            ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputSNGP(
            loss=loss,
            logits=logits,
            cov_matrix=cov_matrix,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ElectraClassificationHeadCustomSNGP(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other, config, ue_config, sn_value=0.95):
        super().__init__()
        self.use_paper_version = ue_config.get("use_paper_version", False)
        self.dropout1 = other.dropout
        if self.use_paper_version:
            self.dense = pt_spectral_norm(other.dense)
        else:
            self.dense = spectral_norm(other.dense, sn_value)
        self.out_proj = GPOutputLayer(config, ue_config)

    def forward(
        self, features, is_final_epoch, is_first_minibatch, is_last_minibatch, epoch
    ):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        if self.use_paper_version:
            x = self.dropout1(x)
        x = self.dense(x)
        x = get_activation("gelu")(
            x
        )
        if not self.use_paper_version:
            x = self.dropout1(x)
        x = self.out_proj(
            x, is_final_epoch, is_first_minibatch, is_last_minibatch, epoch
        )
        return x


class SNGPElectraForSequenceClassificationCached(
    CachedInferenceMixin, ElectraForSequenceClassification
):
    def __init__(self, config, ue_config):
        super().__init__(config)
        self.classifier = ElectraClassificationHeadCustomSNGP(
            self.classifier, config, ue_config
        )
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_final_epoch=False,
        is_first_minibatch=False,
        is_last_minibatch=False,
        epoch=0,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.inference_body(
            self.electra,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        logits, cov_matrix = self.classifier(
            sequence_output,
            is_final_epoch,
            is_first_minibatch,
            is_last_minibatch,
            epoch,
        )

        loss = None

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (
                logits,
                cov_matrix,
            ) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputSNGP(
            loss=loss,
            logits=logits,
            cov_matrix=cov_matrix,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

    

class ElectraNERHeadCustomSNGP(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other, config, ue_config, sn_value=0.95):
        super().__init__()
        self.use_paper_version = ue_config.get("use_paper_version", False)
        if self.use_paper_version:
            self.dense = pt_spectral_norm(nn.Linear(768, 768))
        else:
            self.dense = spectral_norm(nn.Linear(768, 768), sn_value=0.95)
        self.dropout2 = copy.deepcopy(other.dropout)       
        self.out_proj = GPOutputLayer(config, ue_config)
        
    def forward(
        self, features, is_final_epoch, is_first_minibatch, is_last_minibatch, epoch
    ):
        x = features[:, :, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout2(x)
        x = self.dense(x)
        x = get_activation("gelu")(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x_shape = x.shape
        x = x.reshape(-1, x_shape[2])
        x, cov = self.out_proj(
            x, is_final_epoch, is_first_minibatch, is_last_minibatch, epoch
        )
        x = x.reshape(x_shape[0], x_shape[1], -1)
        return (x, torch.abs(cov))

class SNGPElectraForTokenClassificationCached(
    CachedInferenceMixin, ElectraForTokenClassification
):
    def __init__(self, config, ue_config):
        super().__init__(config)
        self.num_labels = len(config.id2label)
        self.classifier = ElectraNERHeadCustomSNGP(
            self, config, ue_config
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_final_epoch=False,
        is_first_minibatch=False,
        is_last_minibatch=False,
        epoch=0,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        discriminator_hidden_states = self.inference_body(
            self.electra,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = discriminator_hidden_states[0]
        logits, cov_matrix = self.classifier(
            sequence_output,
            is_final_epoch,
            is_first_minibatch,
            is_last_minibatch,
            epoch,
        )

        loss = None

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (
                logits,
                cov_matrix,
            ) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputSNGP(
            loss=loss,
            logits=logits,
            cov_matrix=cov_matrix,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )