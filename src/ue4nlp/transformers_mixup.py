"""Models with MSD"""
from typing import Optional

import torch
from torch.nn import CrossEntropyLoss, KLDivLoss, MSELoss

from transformers import (
    BertForTokenClassification,
    BertForSequenceClassification,
    ElectraForTokenClassification,
    ElectraForSequenceClassification,
    XLNetForSequenceClassification,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DebertaForSequenceClassification,
    DebertaForTokenClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.xlnet.modeling_xlnet import XLNetForSequenceClassificationOutput
from .transformers_cached import CachedInferenceMixin


class SeqMSD:
    def __init__(self, *args, **kwargs):
        """Additional class, implements mixup for MSD models for sequence classification.
        """
        super().__init__(*args, **kwargs)

    def post_init(self, mixup):
        self.mixup = mixup.mixup
        self.self_ensembling = mixup.self_ensembling
        self.omega = mixup.omega
        self.lam1 = mixup.lam1
        self.lam2 = mixup.lam2
        if self.self_ensembling:
            self.add_self_ensemble()

    def _mixup(self, pooled_output, labels, pooled_output_2):
        batch_size = pooled_output.shape[0]
        device = pooled_output.device
        ids = torch.randperm(batch_size).view(batch_size, 1)
        ids_onehot = torch.zeros(batch_size, batch_size).scatter_(1, ids, 1).to(device)
        alpha = (
            (torch.rand(batch_size) * (1 - self.omega) + self.omega)
            .unsqueeze(1)
            .unsqueeze(1)
            .to(device)
        )
        pooled_output = (1 - alpha) * pooled_output + alpha * torch.index_select(
            pooled_output, 0, ids.squeeze().long().to(device)
        )
        if pooled_output_2 is not None:
            pooled_output_2 = (
                1 - alpha
            ) * pooled_output_2 + alpha * torch.index_select(
                pooled_output_2, 0, ids.squeeze().long().to(device)
            )
        # cast labels to onehot tensor
        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)
        alpha = alpha.squeeze(1)
        labels = (1 - alpha) * labels.float() + alpha * torch.mm(
            ids_onehot, labels.float()
        )
        return pooled_output, labels, pooled_output_2

    def calc_msd_loss(self, labels, logits, logits_2=None):
        loss = None
        eps = 1e-8
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.mixup and not self.self_ensembling and self.training:
                # use KL loss
                loss_fct = KLDivLoss(reduction="batchmean")
                # use log probas instead of probas
                loss = loss_fct(
                    torch.log(torch.nn.functional.softmax(logits) + eps), labels
                )
            elif self.self_ensembling and self.training:
                # again use KL loss, but add some regularization terms
                loss_fct = KLDivLoss(reduction="batchmean")
                loss_mse = MSELoss()
                labels_2 = labels.clone()
                l1 = logits.clone()
                l2 = logits_2.clone()
                # use log probas instead of probas - get probas with softmax, after calc log probas with eps for numerical stable log
                loss_1 = loss_fct(torch.log(torch.nn.functional.softmax(logits, dim=-1) + eps), labels)
                loss_2 = loss_fct(torch.log(torch.nn.functional.softmax(logits_2, dim=-1) + eps), labels_2)
                loss_3 = loss_mse(l1.view(-1), l2.view(-1))
                loss = loss_1 + self.lam1 * loss_2 + self.lam2 * loss_3
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss


class TokenMSD:
    def __init__(self, *args, **kwargs):
        """Additional class, implements mixup for MSD models for token classification.
        """
        super().__init__(*args, **kwargs)

    def post_init(self, mixup):
        self.mixup = mixup.mixup
        self.self_ensembling = mixup.self_ensembling
        self.omega = mixup.omega
        self.lam1 = mixup.lam1
        self.lam2 = mixup.lam2
        if self.self_ensembling:
            self.add_self_ensemble()

    def _mixup(self, pooled_output, labels, pooled_output_2):
        batch_size = pooled_output.shape[0]
        seq_size = pooled_output.shape[1]
        device = pooled_output.device
        # For NER we iterate by batches to ensure that tensors fit into memory
        new_labels = torch.zeros(batch_size, seq_size, self.num_labels).to(device)
        for i in range(batch_size):
            # use only indices with non-padding labels
            ids_to_use = torch.where(labels[i] != -100)[0].to(device)
            # shuffled indices
            ids = ids_to_use[torch.randperm(len(ids_to_use))].to(device)
            alpha = (
                (torch.rand(len(ids)) * (1 - self.omega) + self.omega)
                .unsqueeze(1)
                .to(device)
            )
            pooled_output[i][ids_to_use] = (1 - alpha) * pooled_output[i][ids_to_use] + alpha * pooled_output[i][ids]
            if pooled_output_2 is not None:
                pooled_output_2[i][ids_to_use] = (1 - alpha) * pooled_output_2[i][ids_to_use] + alpha * pooled_output_2[i][ids]
            # replace padding labels with 0 label to use one-hot encoding
            padding_ids = labels[i] == -100
            labels[i][padding_ids] = 0
            new_labels[i] = torch.nn.functional.one_hot(
                labels[i], num_classes=self.num_labels
            )
            # after replace all padding labels with 0
            new_labels[i][ids_to_use] = (1 - alpha) * new_labels[i][ids_to_use].float() + alpha * new_labels[i][ids].float()
            new_labels[i][padding_ids] = 0
        return pooled_output, new_labels, pooled_output_2

    def calc_msd_loss(self, labels, logits, padding_ids, attention_mask=None, logits_2=None):
        loss = None
        eps = 1e-8
        if labels is not None:
            if attention_mask is not None:
                # ignore loss not only on padding ids, but also on 0 attention mask
                padding_ids = padding_ids * (attention_mask.view(-1) == 1).squeeze()
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.mixup and not self.self_ensembling and self.training:
                # use KL loss
                loss_fct = KLDivLoss(reduction="batchmean")
                # use log probas instead of probas
                logits = logits.reshape(-1, self.num_labels)
                labels = labels.reshape(-1, self.num_labels)
                loss = loss_fct(torch.log(torch.nn.functional.softmax(logits[padding_ids], dim=-1) + eps), labels[padding_ids])
            elif self.self_ensembling and self.training:
                logits = logits.reshape(-1, self.num_labels)
                labels = labels.reshape(-1, self.num_labels)
                logits_2 = logits_2.reshape(-1, self.num_labels)
                # again use KL loss, but add some regularization terms
                # here we set reduction to None, because we want to calc loss only on attention_mask==1
                loss_fct = KLDivLoss(reduction="batchmean")
                loss_mse = MSELoss()
                # clone tensors for diff losses
                labels_2 = labels.clone()
                l1 = logits.clone()
                l2 = logits_2.clone()
                # use log probas instead of probas - get probas with softmax, after calc log probas with eps for numerical stable log
                loss_1 = loss_fct(torch.log(torch.nn.functional.softmax(logits[padding_ids], dim=-1) + eps), labels[padding_ids])
                loss_2 = loss_fct(torch.log(torch.nn.functional.softmax(logits_2[padding_ids], dim=-1) + eps), labels_2[padding_ids])
                loss_3 =  loss_mse(l1[padding_ids].view(-1), l2[padding_ids].view(-1))
                loss = loss_1 + self.lam1 * loss_2 + self.lam2 * loss_3
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss


class ElectraForSequenceClassificationMSD(SeqMSD, ElectraForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def add_self_ensemble(self):
        """Adds another network for self-ensembling"""
        self.model_2 = ElectraForSequenceClassification(self.config)

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
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        logits_2 = None

        discriminator_hidden_states = self.electra(
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
        # add MixUp - only on train, then we have labels
        if (
            self.mixup
            and labels is not None
            and not self.self_ensembling
            and self.training
        ):
            # also add fc0 and dropout, as in paper
            sequence_output, labels, _ = self._mixup(sequence_output, labels, None)
        if self.self_ensembling and self.training:
            outputs_2 = self.model_2.electra(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output_2 = outputs_2[0]

            if self.mixup and labels is not None and self.training:
                sequence_output, labels, pooled_output_2 = self._mixup(
                    sequence_output, labels, pooled_output_2
                )
            elif not(self.mixup) and labels is not None and self.training:
                labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()
            logits_2 = self.model_2.classifier(pooled_output_2)
        logits = self.classifier(sequence_output)

        loss = None
        eps = 1e-8
        if labels is not None:
            loss = self.calc_msd_loss(labels, logits, logits_2)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class ElectraForTokenClassificationMSD(TokenMSD, ElectraForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = self.config.num_labels

    def add_self_ensemble(self):
        """Adds another network for self-ensembling"""
        self.model_2 = ElectraForTokenClassification(self.config)

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
    ):

        # hotfix
        if str(self.electra.device) != 'cpu' and str(input_ids.device) == 'cpu':
            input_ids = input_ids.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )

        discriminator_sequence_output = discriminator_hidden_states[0]
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits_2 = None
        # get padding indices
        if labels is not None:
            padding_ids = (labels.view(-1) != -100).squeeze()
        if (
            self.mixup
            and labels is not None
            and not self.self_ensembling
            and self.training
        ):
            # also add fc0 and dropout, as in paper
            discriminator_sequence_output, labels, _ = self._mixup(
                discriminator_sequence_output, labels, None
            )
        if self.self_ensembling and self.training:
            outputs_2 = self.model_2.electra(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            pooled_output_2 = outputs_2[0]
            # TODO: replace on model_2.dropout and rerun
            pooled_output_2 = self.model_2.dropout(pooled_output_2)

            if self.mixup and labels is not None and self.training:
                discriminator_sequence_output, labels, pooled_output_2 = self._mixup(
                    discriminator_sequence_output, labels, pooled_output_2
                )
            elif not(self.mixup) and labels is not None and self.training:
                labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()
            logits_2 = self.model_2.classifier(pooled_output_2)
        logits = self.classifier(discriminator_sequence_output)
        # print(logits.shape, self.num_labels)
        # print(torch.unique(labels))
        output = (logits,)

        loss = None
        eps = 1e-8
        if labels is not None:
            loss = self.calc_msd_loss(labels, logits, padding_ids, attention_mask, logits_2)
            output = (loss,) + output

        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)


class XLNetForSequenceClassificationMSD(XLNetForSequenceClassification):
    # Implement XLNet with MSD for tests
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def add_self_ensemble(self):
        """Adds another network for self-ensembling"""
        self.model_2 = XLNetForSequenceClassification(self.config)

    def post_init(self, mixup):
        self.mixup = mixup.mixup
        self.self_ensembling = mixup.self_ensembling
        self.omega = mixup.omega
        self.lam1 = mixup.lam1
        self.lam2 = mixup.lam2
        if self.self_ensembling:
            self.add_self_ensemble()
        # now we add freezing of all layers, as in paper
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.model_2.transformer.parameters():
            param.requires_grad = False
        print('Freezed')

    def _mixup(self, pooled_output, labels, pooled_output_2):
        batch_size = pooled_output.shape[0]
        device = pooled_output.device
        ids = torch.randperm(batch_size).view(batch_size, 1)
        ids_onehot = torch.zeros(batch_size, batch_size).scatter_(1, ids, 1).to(device)
        alpha = (
            (torch.rand(batch_size) * (1 - self.omega) + self.omega)
            .unsqueeze(1)
            .unsqueeze(1)
            .to(device)
        )
        pooled_output = (1 - alpha) * pooled_output + alpha * torch.index_select(
            pooled_output, 0, ids.squeeze().long().to(device)
        )
        if pooled_output_2 is not None:
            pooled_output_2 = (
                1 - alpha
            ) * pooled_output_2 + alpha * torch.index_select(
                pooled_output_2, 0, ids.squeeze().long().to(device)
            )
        # cast labels to onehot tensor
        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)
        alpha = alpha.squeeze(1)
        labels = (1 - alpha) * labels.float() + alpha * torch.mm(
            ids_onehot, labels.float()
        )
        return pooled_output, labels, pooled_output_2

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        sequence_output = transformer_outputs[0]

        # add MixUp - only on train, then we have labels
        if (
            self.mixup
            and labels is not None
            and not self.self_ensembling
            and self.training
        ):
            # also add fc0 and dropout, as in paper
            sequence_output, labels, _ = self._mixup(sequence_output, labels, None)
        if self.self_ensembling and self.training:
            outputs_2 = self.model_2.transformer(
                input_ids,
                attention_mask=attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                token_type_ids=token_type_ids,
                input_mask=input_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_mems=use_mems,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

            pooled_output_2 = outputs_2[0]

            if self.mixup and labels is not None and self.training:
                sequence_output, labels, pooled_output_2 = self._mixup(
                    sequence_output, labels, pooled_output_2
                )
            elif not(self.mixup) and labels is not None and self.training:
                labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()
            output_2 = self.sequence_summary(pooled_output_2)
            logits_2 = self.logits_proj(output_2)
            # logits_2 = self.model_2.classifier(pooled_output_2)
        output = self.sequence_summary(sequence_output)
        logits = self.logits_proj(output)

        loss = None
        eps = 1e-8
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.mixup and not self.self_ensembling and self.training:
                # use KL loss
                loss_fct = KLDivLoss(reduction="batchmean")
                # use log probas instead of probas
                loss = loss_fct(
                    torch.log(torch.nn.functional.softmax(logits) + eps), labels
                )
            elif self.self_ensembling and self.training:
                # again use KL loss, but add some regularization terms
                loss_fct = KLDivLoss(reduction="batchmean")
                loss_mse = MSELoss()
                # use log probas instead of probas - get probas with softmax, after calc log probas with eps for numerical stable log
                loss = (
                    loss_fct(
                        torch.log(torch.nn.functional.softmax(logits, dim=-1) + eps),
                        labels,
                    )
                    + self.lam1
                    * loss_fct(
                        torch.log(torch.nn.functional.softmax(logits_2, dim=-1) + eps),
                        labels,
                    )
                    + self.lam2 * loss_mse(logits.view(-1), logits_2.view(-1))
                )
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLNetCachedInferenceMixin(CachedInferenceMixin):
    def __init__(self, config):
        super().__init__(config)

    def inference_body(
        self,
        body,
        input_ids,
        attention_mask,
        mems,
        perm_mask,
        target_mapping,
        token_type_ids,
        input_mask,
        head_mask,
        inputs_embeds,
        use_mems,
        output_attentions,
        output_hidden_states,
        return_dict,
        **kwargs,
    ):
        cache_key = self.create_cache_key(input_ids)

        if not self.use_cache or cache_key not in self.cache:
            hidden_states = body(
                input_ids,
                attention_mask=attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                token_type_ids=token_type_ids,
                input_mask=input_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_mems=use_mems,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

            if self.use_cache and (
                self.cache_size is None or len(self.cache) < self.cache_size
            ):
                # added part for tuples - this needed for metric regularizer, because
                # with it we set output_hidden to True
                self.cache[cache_key] = {
                    n: o.detach().cpu()
                    if (o is not None and not isinstance(o, tuple))
                    else o
                    for n, o in hidden_states.__dict__.items()
                }
        else:
            hidden_states = BaseModelOutputWithPoolingAndCrossAttentions(
                **{
                    n: o.cuda() if (o is not None and not isinstance(o, tuple)) else o
                    for n, o in self.cache[cache_key].items()
                }
            )

        return hidden_states

class XLNetForSequenceClassificationCachedMSD(
    XLNetCachedInferenceMixin,XLNetForSequenceClassification):
    # Implement XLNet with MSD for tests
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def add_self_ensemble(self):
        """Adds another network for self-ensembling"""
        self.model_2 = XLNetForSequenceClassification(self.config)

    def post_init(self, mixup):
        self.mixup = mixup.mixup
        self.self_ensembling = mixup.self_ensembling
        self.omega = mixup.omega
        self.lam1 = mixup.lam1
        self.lam2 = mixup.lam2
        if self.self_ensembling and self.training:
            self.add_self_ensemble()

    def _mixup(self, pooled_output, labels, pooled_output_2):
        batch_size = pooled_output.shape[0]
        device = pooled_output.device
        ids = torch.randperm(batch_size).view(batch_size, 1)
        ids_onehot = torch.zeros(batch_size, batch_size).scatter_(1, ids, 1).to(device)
        alpha = (
            (torch.rand(batch_size) * (1 - self.omega) + self.omega)
            .unsqueeze(1)
            .unsqueeze(1)
            .to(device)
        )
        pooled_output = (1 - alpha) * pooled_output + alpha * torch.index_select(
            pooled_output, 0, ids.squeeze().long().to(device)
        )
        if pooled_output_2 is not None:
            pooled_output_2 = (
                1 - alpha
            ) * pooled_output_2 + alpha * torch.index_select(
                pooled_output_2, 0, ids.squeeze().long().to(device)
            )
        # cast labels to onehot tensor
        labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)
        alpha = alpha.squeeze(1)
        labels = (1 - alpha) * labels.float() + alpha * torch.mm(
            ids_onehot, labels.float()
        )
        return pooled_output, labels, pooled_output_2

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mems=None,
        perm_mask=None,
        target_mapping=None,
        token_type_ids=None,
        input_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_mems=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.inference_body(
            self.transformer,
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        # TODO: change cached call to use all XLNet args
        sequence_output = transformer_outputs[0]

        # add MixUp - only on train, then we have labels
        if (
            self.mixup
            and labels is not None
            and not self.self_ensembling
            and self.training
        ):
            # also add fc0 and dropout, as in paper
            sequence_output, labels, _ = self._mixup(sequence_output, labels, None)
        if self.self_ensembling and self.training:
            # Here we doesn't cache hiddens of second model,
            # cause we don't use it on inference
            outputs_2 = self.model_2.transformer(
                input_ids,
                attention_mask=attention_mask,
                mems=mems,
                perm_mask=perm_mask,
                target_mapping=target_mapping,
                token_type_ids=token_type_ids,
                input_mask=input_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_mems=use_mems,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

            pooled_output_2 = outputs_2[0]

            if self.mixup and labels is not None and self.training:
                sequence_output, labels, pooled_output_2 = self._mixup(
                    sequence_output, labels, pooled_output_2
                )
            elif not(self.mixup) and labels is not None and self.training:
                labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()
            output_2 = self.sequence_summary(pooled_output_2)
            logits_2 = self.logits_proj(output_2)
            # logits_2 = self.model_2.classifier(pooled_output_2)
        output = self.sequence_summary(sequence_output)
        logits = self.logits_proj(output)

        loss = None
        eps = 1e-8
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.mixup and not self.self_ensembling and self.training:
                # use KL loss
                loss_fct = KLDivLoss(reduction="batchmean")
                # use log probas instead of probas
                loss = loss_fct(
                    torch.log(torch.nn.functional.softmax(logits) + eps), labels
                )
            elif self.self_ensembling and self.training:
                # again use KL loss, but add some regularization terms
                loss_fct = KLDivLoss(reduction="batchmean")
                loss_mse = MSELoss()
                # use log probas instead of probas - get probas with softmax, after calc log probas with eps for numerical stable log
                loss = (
                    loss_fct(
                        torch.log(torch.nn.functional.softmax(logits, dim=-1) + eps),
                        labels,
                    )
                    + self.lam1
                    * loss_fct(
                        torch.log(torch.nn.functional.softmax(logits_2, dim=-1) + eps),
                        labels,
                    )
                    + self.lam2 * loss_mse(logits.view(-1), logits_2.view(-1))
                )
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class DistilBertForSequenceClassificationMSD(
    SeqMSD, DistilBertForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def add_self_ensemble(self):
        """Adds another network for self-ensembling"""
        self.model_2 = DistilBertForSequenceClassification(self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        logits_2 = None

        distilbert_output = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        # add MixUp - only on train, then we have labels
        if (
            self.mixup
            and labels is not None
            and not self.self_ensembling
            and self.training
        ):
            # also add fc0 and dropout, as in paper
            hidden_state, labels, _ = self._mixup(hidden_state, labels, None)
        if self.self_ensembling and self.training:
            distilbert_output_2 = self.model_2.distilbert(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_state_2 = distilbert_output_2[0]

            if self.mixup and labels is not None and self.training:
                hidden_state, labels, hidden_state_2 = self._mixup(
                    hidden_state, labels, hidden_state_2
                )
            elif not(self.mixup) and labels is not None and self.training:
                labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()
            pooled_output_2 = hidden_state_2[:, 0]  # (bs, dim)
            pooled_output_2 = self.model_2.pre_classifier(pooled_output_2)  # (bs, dim)
            pooled_output_2 = torch.nn.ReLU()(pooled_output_2)  # (bs, dim)
            pooled_output_2 = self.model_2.dropout(pooled_output_2)  # (bs, dim)
            logits_2 = self.model_2.classifier(pooled_output_2)  # (bs, num_labels)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = torch.nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        eps = 1e-8
        if labels is not None:
            loss = self.calc_msd_loss(labels, logits, logits_2)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class DistilBertForTokenClassificationMSD(TokenMSD, DistilBertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def add_self_ensemble(self):
        """Adds another network for self-ensembling"""
        self.model_2 = DistilBertForTokenClassification(self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        # hotfix
        if str(self.distilbert.device) != 'cpu' and str(input_ids.device) == 'cpu':
            input_ids = input_ids.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits_2 = None
        # get padding indices
        if labels is not None:
            padding_ids = (labels.view(-1) != -100).squeeze()
        if (
            self.mixup
            and labels is not None
            and not self.self_ensembling
            and self.training
        ):
            # also add fc0 and dropout, as in paper
            sequence_output, labels, _ = self._mixup(
                sequence_output, labels, None
            )
        if self.self_ensembling and self.training:
            outputs_2 = self.model_2.distilbert(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            sequence_output_2 = outputs_2[0]
            sequence_output_2 = self.model_2.dropout(sequence_output_2)

            if self.mixup and labels is not None and self.training:
                sequence_output, labels, sequence_output_2 = self._mixup(
                    sequence_output, labels, sequence_output_2
                )
            elif not(self.mixup) and labels is not None and self.training:
                labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()
            logits_2 = self.model_2.classifier(sequence_output_2)
        logits = self.classifier(sequence_output)

        loss = None
        eps = 1e-8
        outputs = (logits,) + outputs[
            1:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            loss = self.calc_msd_loss(labels, logits, padding_ids, attention_mask, logits_2)
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class DebertaForSequenceClassificationMSD(
    SeqMSD, DebertaForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def add_self_ensemble(self):
        """Adds another network for self-ensembling"""
        self.model_2 = DebertaForSequenceClassification(self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        logits_2 = None

        discriminator_hidden_states = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        encoder_layer = discriminator_hidden_states[0]

        
        # add MixUp - only on train, then we have labels
        if (
            self.mixup
            and labels is not None
            and not self.self_ensembling
            and self.training
        ):
            # also add fc0 and dropout, as in paper
            encoder_layer, labels, _ = self._mixup(encoder_layer, labels, None)
        if self.self_ensembling and self.training:
            discriminator_hidden_states_2 = self.model_2.deberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            encoder_layer_2 = discriminator_hidden_states_2[0]

            if self.mixup and labels is not None and self.training:
                encoder_layer, labels, encoder_layer_2 = self._mixup(
                    encoder_layer, labels, encoder_layer_2
                )
            elif not(self.mixup) and labels is not None and self.training:
                labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()
            pooled_output_2 = self.model_2.pooler(encoder_layer_2)
            pooled_output_2 = self.model_2.dropout(pooled_output_2)
            logits_2 = self.model_2.classifier(pooled_output_2)
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        eps = 1e-8
        if labels is not None:
            loss = self.calc_msd_loss(labels, logits, logits_2)

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class DebertaForTokenClassificationMSD(TokenMSD, DebertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def add_self_ensemble(self):
        """Adds another network for self-ensembling"""
        self.model_2 = DebertaForTokenClassification(self.config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # hotfix
        if str(self.deberta.device) != 'cpu' and str(input_ids.device) == 'cpu':
            input_ids = input_ids.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
        discriminator_hidden_states = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = discriminator_hidden_states[0]
        sequence_output = self.dropout(sequence_output)
        logits_2 = None
        # get non-padding indices
        if labels is not None:
            padding_ids = (labels.view(-1) != -100).squeeze()
        if (
            self.mixup
            and labels is not None
            and not self.self_ensembling
            and self.training
        ):
            # also add fc0 and dropout, as in paper
            sequence_output, labels, _ = self._mixup(
                sequence_output, labels, None
            )
        if self.self_ensembling and self.training:
            discriminator_hidden_states_2 = self.model_2.deberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output_2 = discriminator_hidden_states_2[0]
            sequence_output_2 = self.model_2.dropout(sequence_output_2)

            if self.mixup and labels is not None and self.training:
                sequence_output, labels, sequence_output_2 = self._mixup(
                    sequence_output, labels, sequence_output_2
                )
            elif not(self.mixup) and labels is not None and self.training:
                labels = torch.nn.functional.one_hot(labels, num_classes=self.num_labels).float()
            logits_2 = self.model_2.classifier(sequence_output_2)
        logits = self.classifier(sequence_output)
        output = (logits,)

        if labels is not None:
            loss = self.calc_msd_loss(labels, logits, padding_ids, attention_mask, logits_2)
            output = (loss,) + output
        output += discriminator_hidden_states[1:]
        return output  # (loss), scores, (hidden_states), (attentions)
