from typing import Optional

import torch
import torch.nn as nn
from torch.nn import (
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    MSELoss,
)

from transformers import (
    BertForTokenClassification,
    BertForSequenceClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification,
    ElectraForSequenceClassification,
    RobertaForSequenceClassification,
    DebertaForSequenceClassification,
    DebertaForTokenClassification,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
)


class BertForTokenClassificationCached(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.use_cache = False
        self.cache_size = None
        self.cache = dict()

    def empty_cache(self):
        self.cache.clear()

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.use_cache = False
        self.cache_size = None
        self.empty_cache()

    def set_cache_size(self, size: Optional[int] = 25):
        self.cache_size = size

    @staticmethod
    def create_cache_key(tensor: torch.Tensor) -> int:
        return hash(frozenset(tensor.cpu().numpy().ravel()))

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

        cache_key = self.create_cache_key(input_ids)
        if not self.use_cache or cache_key not in self.cache:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if self.use_cache and (
                self.cache_size is None or len(self.cache) < self.cache_size
            ):
                # print('WRITING TO CACHE')
                # self.cache[cache_key] = outputs
                self.cache[cache_key] = tuple(o.detach().cpu() for o in outputs)
        else:
            # print('USING CACHED OUTPUTS')
            # outputs = self.cache[cache_key]
            outputs = tuple(o.cuda() for o in self.cache[cache_key])

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            2:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class DistilBertForTokenClassificationCached(DistilBertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.use_cache = False
        self.cache_size = None
        self.cache = dict()

    def empty_cache(self):
        self.cache.clear()

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.use_cache = False
        self.empty_cache()

    def set_cache_size(self, size: Optional[int] = 25):
        self.cache_size = size

    @staticmethod
    def create_cache_key(tensor: torch.Tensor) -> int:
        return hash(frozenset(tensor.cpu().numpy().ravel()))

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

        cache_key = self.create_cache_key(input_ids)
        if not self.use_cache or cache_key not in self.cache:
            outputs = self.distilbert(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if self.use_cache and (
                self.cache_size is None or len(self.cache) < self.cache_size
            ):
                # print('WRITING TO CACHE')
                # self.cache[cache_key] = outputs
                self.cache[cache_key] = tuple(o.detach().cpu() for o in outputs)

        else:
            # print('USING CACHED OUTPUTS')
            # outputs = self.cache[cache_key]
            outputs = tuple(o.cuda() for o in self.cache[cache_key])

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            1:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class ElectraForTokenClassificationCached(ElectraForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.use_cache = False
        self.cache_size = None
        self.cache = dict()

    def empty_cache(self):
        self.cache.clear()

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.use_cache = False
        self.empty_cache()

    def set_cache_size(self, size: Optional[int] = 1000):
        self.cache_size = size

    @staticmethod
    def create_cache_key(tensor: torch.Tensor) -> int:
        return hash(frozenset(tensor.cpu().numpy().ravel()))

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
        cache_key = self.create_cache_key(input_ids)

        if not self.use_cache or cache_key not in self.cache:
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
            if self.use_cache and (
                self.cache_size is None or len(self.cache) < self.cache_size
            ):
                # print('WRITING TO CACHE')
                # self.cache[cache_key] = discriminator_hidden_states
                self.cache[cache_key] = tuple(
                    discriminator_hidden_states[o].detach().cpu()
                    for o in discriminator_hidden_states
                )
        else:
            # print('USING CACHED OUTPUTS')
            discriminator_hidden_states = tuple(o.cuda() for o in self.cache[cache_key])

        discriminator_sequence_output = discriminator_hidden_states[0]
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)

        output = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels), labels.view(-1)
                )

            output = (loss,) + output

        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)


class CachedInferenceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_cache = False
        self.cache_size = None
        self.cache = dict()

    def empty_cache(self):
        self.cache.clear()

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.use_cache = False
        self.empty_cache()

    def set_cache_size(self, size: Optional[int] = 25):
        self.cache_size = size

    @staticmethod
    def create_cache_key(tensor: torch.Tensor) -> int:
        return hash(frozenset(tensor.cpu().numpy().ravel()))

    def inference_body(
        self,
        body,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        head_mask,
        inputs_embeds,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        cache_key = self.create_cache_key(input_ids)

        if not self.use_cache or cache_key not in self.cache:
            if head_mask is not None:
                hidden_states = body(
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
            else:
                hidden_states = body(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
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


class ElectraForSequenceClassificationCached(
    CachedInferenceMixin, ElectraForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)

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
        logits = self.classifier(sequence_output)

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
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )


class BertForSequenceClassificationCached(
    CachedInferenceMixin, BertForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)

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

        outputs = self.inference_body(
            body=self.bert,
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForSequenceClassificationCached(
    CachedInferenceMixin, RobertaForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)

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

        discriminator_hidden_states = self.inference_body(
            self.roberta,
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
        logits = self.classifier(sequence_output)

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
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
    
    
class DebertaForSequenceClassificationCached(
    CachedInferenceMixin, DebertaForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)

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

        discriminator_hidden_states = self.inference_body(
            self.deberta,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        encoder_layer = discriminator_hidden_states[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
    
    
class DebertaForTokenClassificationCached(CachedInferenceMixin, DebertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)

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

        discriminator_hidden_states = self.inference_body(
            self.deberta,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
      
        sequence_output = discriminator_hidden_states[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        output = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels), labels.view(-1)
                )

            output = (loss,) + output
            
        output += discriminator_hidden_states[1:]

        return output  # (loss), scores, (hidden_states), (attentions)

    
class DistilBertCachedInferenceMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_cache = False
        self.cache_size = None
        self.cache = dict()

    def empty_cache(self):
        self.cache.clear()

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.use_cache = False
        self.empty_cache()

    def set_cache_size(self, size: Optional[int] = 25):
        self.cache_size = size

    @staticmethod
    def create_cache_key(tensor: torch.Tensor) -> int:
        return hash(frozenset(tensor.cpu().numpy().ravel()))

    def inference_body(
        self,
        body,
        input_ids,
        attention_mask,
        head_mask,
        inputs_embeds,
        output_attentions,
        output_hidden_states,
        return_dict,
    ):
        cache_key = self.create_cache_key(input_ids)

        if not self.use_cache or cache_key not in self.cache:
            hidden_states = body(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
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


class DistilBertForTokenClassificationCached(DistilBertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.use_cache = False
        self.cache_size = None
        self.cache = dict()

    def empty_cache(self):
        self.cache.clear()

    def enable_cache(self):
        self.use_cache = True

    def disable_cache(self):
        self.use_cache = False
        self.empty_cache()

    def set_cache_size(self, size: Optional[int] = 25):
        self.cache_size = size

    @staticmethod
    def create_cache_key(tensor: torch.Tensor) -> int:
        return hash(frozenset(tensor.cpu().numpy().ravel()))

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

        cache_key = self.create_cache_key(input_ids)
        if not self.use_cache or cache_key not in self.cache:
            outputs = self.distilbert(
                input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            if self.use_cache and (
                self.cache_size is None or len(self.cache) < self.cache_size
            ):
                # print('WRITING TO CACHE')
                # self.cache[cache_key] = outputs
                self.cache[cache_key] = tuple(outputs[o].detach().cpu() for o in outputs)

        else:
            # print('USING CACHED OUTPUTS')
            # outputs = self.cache[cache_key]
            outputs = tuple(o.cuda() for o in self.cache[cache_key])

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            1:
        ]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)



class DistilBertForSequenceClassificationCached(
    DistilBertCachedInferenceMixin, DistilBertForSequenceClassification
):
    def __init__(self, config):
        super().__init__(config)

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

        distilbert_output = self.inference_body(
            self.distilbert,
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
    
    

class ElectraForSequenceClassificationAllLayers(ElectraForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

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
        output_hidden_states=True,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
        
        features = discriminator_hidden_states[0]
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        hidden_states = [hs.detach().cpu()[:, 0, :] for hs in discriminator_hidden_states.hidden_states]
        discriminator_hidden_states.hidden_states = hidden_states
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )