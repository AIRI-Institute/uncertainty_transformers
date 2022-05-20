import torch.nn as nn
from transformers.activations import get_activation
import copy
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as pt_spectral_norm
from utils.spectral_norm import spectral_norm
from typing import List, Optional, Tuple, Union
import math


class ElectraClassificationHeadCustom(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other):
        super().__init__()
        self.dropout1 = other.dropout
        self.dense = other.dense
        self.dropout2 = copy.deepcopy(other.dropout)
        self.out_proj = other.out_proj

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout1(x)
        x = self.dense(x)
        x = get_activation("gelu")(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x


class ElectraClassificationHeadSN(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other, sn_value=None, n_power_iterations=1):
        super().__init__()
        self.dropout1 = other.dropout
        if sn_value is None:
            self.dense = pt_spectral_norm(other.dense, n_power_iterations=n_power_iterations)
        else:
            self.dense = spectral_norm(other.dense, sn_value=sn_value)
        self.dropout2 = copy.deepcopy(other.dropout)
        self.out_proj = other.out_proj
        
    def eval_init(self, other):
        """This function is used for loading model from saved checkpoint, then we have to load CustomHead too"""
        # Here we load weights of already initialized model, so we don't do spectral_norm
        # also add some weights, that changed after spectral_norm call
        self.dense.weight_orig.data = other["classifier.dense.weight_orig"].data
        self.dense.weight_u.data = other["classifier.dense.weight_u"].data
        self.dense.weight_v.data = other["classifier.dense.weight_v"].data
        self.dense.bias.data = other["classifier.dense.bias"].data
        self.out_proj.weight.data = other["classifier.out_proj.weight"].data
        self.out_proj.bias.data = other["classifier.out_proj.bias"].data

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout1(x)
        x = self.dense(x)
        x = get_activation("gelu")(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x


class ElectraNERHeadSN(nn.Module):
    """Head for token-level classification tasks."""

    def __init__(self, other, sn_value=None, n_power_iterations=1):
        super().__init__()
        if sn_value is None:
            self.linear = pt_spectral_norm(nn.Linear(768, 768), n_power_iterations=n_power_iterations)
        else:
            self.linear = spectral_norm(nn.Linear(768, 768), sn_value=sn_value)
        self.dropout1 = copy.deepcopy(other.dropout)
        self.dropout2 = copy.deepcopy(other.dropout)
        self.out_proj = other.classifier

    def eval_init(self, other):
        """This function is used for loading model from saved checkpoint, then we have to load CustomHead too"""
        # Here we load weights of already initialized model, so we don't do spectral_norm
        # also add some weights, that changed after spectral_norm call
        self.linear.weight_orig.data = other["classifier.linear.weight_orig"].data
        self.linear.weight_u.data = other["classifier.linear.weight_u"].data
        self.linear.weight_v.data = other["classifier.linear.weight_v"].data
        self.linear.bias.data = other["classifier.linear.bias"].data
        self.out_proj.weight.data = other["classifier.out_proj.weight"].data
        self.out_proj.bias.data = other["classifier.out_proj.bias"].data

    def forward(self, features, **kwargs):
        x = features[:, :, :]  # take all
        x = self.dropout1(x)
        x = self.linear(x)
        x = get_activation("gelu")(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x


class BertClassificationHeadIdentityPooler(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other):
        super().__init__()

    def forward(self, pooled_output):
        return pooled_output  # , torch.rand((pooled_output.shape[0], pooled_output.shape[0]))


class ElectraClassificationHeadIdentityPooler(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other, use_activation=True):
        super().__init__()
        self.use_activation = use_activation
        self.dropout1 = other.dropout1
        self.dense = other.dense
        # self.dropout2 = copy.deepcopy(other.dropout)
        # self.out_proj = other.out_proj

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout1(x)
        x = self.dense(x)
        if self.use_activation:
            x = get_activation("gelu")(
                x
            )
        return x  # , torch.rand((pooled_output.shape[0], pooled_output.shape[0]))


class XLNetClassificationHeadIdentityPooler(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, features):
        return features


class ElectraNERHeadCustom(nn.Module):
    """Head for token-level classification tasks."""

    def __init__(self, other):
        super().__init__()
        self.linear = nn.Linear(768, 768)
        self.dropout1 = copy.deepcopy(other.dropout)
        self.dropout2 = copy.deepcopy(other.dropout)
        self.out_proj = other.classifier

    def eval_init(self, other):
        """This function is used for loading model from saved checkpoint, then we have to load CustomHead too"""
        try:
            if "classifier.out_proj.linear.weight" in other.keys():
                self.linear.weight.data = other["classifier.out_proj.linear.weight"].data
            else:
                self.linear.weight.data = other["classifier.out_proj.linear.weight_orig"].data
            self.linear.bias.data = other["classifier.out_proj.linear.bias"].data
            self.out_proj.weight.data = other["classifier.out_proj.out_proj.weight"].data
            self.out_proj.bias.data = other["classifier.out_proj.out_proj.bias"].data
        except:
            if "classifier.linear.weight" in other.keys():
                self.linear.weight.data = other["classifier.linear.weight"].data
            else:
                self.linear.weight.data = other["classifier.linear.weight_orig"].data
            self.linear.bias.data = other["classifier.linear.bias"].data
            self.out_proj.weight.data = other["classifier.out_proj.weight"].data
            self.out_proj.bias.data = other["classifier.out_proj.bias"].data

    def forward(self, features, **kwargs):
        x = features[:, :, :]  # take all
        x = self.dropout1(x)
        x = self.linear(x)
        x = get_activation("gelu")(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x


class ElectraNERHeadIdentityPooler(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other, use_activation=True):
        super().__init__()
        self.use_activation = use_activation
        self.linear = other.linear
        # self.dropout2 = copy.deepcopy(other.dropout)
        # self.out_proj = other.out_proj

    def forward(self, features):
        x = features[:, :, :]  # take all
        x = self.linear(x)
        if self.use_activation:
            x = get_activation("gelu")(
                x
            )
        return x


def spectral_normalized_model(
    model: torch.nn.Module, substitution_layer_names=["ElectraOutput"]
):
    for i, layer in enumerate(list(model.children())):
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name in substitution_layer_names:
            model._modules[module_name].dense = spectral_norm(
                model._modules[module_name].dense
            )
        else:
            spectral_normalized_model(model=layer)
            
class SpectralNormalizedPooler(torch.nn.Module):
    def __init__(self, pooler, sn_value=None):
        super().__init__()
        if sn_value is None:
            self.dense = pt_spectral_norm(
                pooler.dense
            )
        else:
            self.dense = spectral_norm(
                pooler.dense, sn_value
            )
            
        self.dropout = pooler.dropout
        self.config = pooler.config
        
    def eval_init(self, other):
        """This function is used for loading model from saved checkpoint, then we have to load CustomHead too"""
        # Here we load weights of already initialized model, so we don't do spectral_norm
        # also add some weights, that changed after spectral_norm call
        self.dense.weight_orig.data = other["pooler.dense.weight_orig"].data
        self.dense.weight_u.data = other["pooler.dense.weight_u"].data
        self.dense.weight_v.data = other["pooler.dense.weight_v"].data
        self.dense.bias.data = other["pooler.dense.bias"].data

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = get_activation(self.config.pooler_hidden_act)(pooled_output)
        return pooled_output


class ElectraSelfAttentionStochastic(nn.Module):
    def __init__(self, config, position_embedding_type=None, hierarchial=False, num_centroids=16, tau_1=1, tau_2=1):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.hierarchial = hierarchial
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.num_centroids = num_centroids
        if self.hierarchial:
            self.centroids = nn.Linear(self.attention_head_size, self.num_centroids, bias=False)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def post_init(self, other):
        # copy weights of other
        if isinstance(other, ElectraSelfAttentionStochastic) and self.hierarchial and other.hierarchial:
            # copy centroids
            self.centroids.weight.data = other.centroids.weight.data
        for layer in ["query", "key", "value"]:
            getattr(self, layer).weight.data = getattr(other, layer).weight.data
            getattr(self, layer).bias.data = getattr(other, layer).bias.data

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def calc_stochastic_attention(attention_scores, tau=1):
        # draw samples from distribution
        samples = nn.functional.gumbel_softmax(attention_scores)
        # calc softmax
        probas = nn.functional.softmax((attention_scores + samples) / tau, dim=-1)
        return probas

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if self.hierarchial:
            # calc a_c with gumbel-softmax
            attention_centroids = self.calc_stochastic_attention(self.centroids(key_layer), self.tau_1)
            key_centroids = torch.matmul(attention_centroids, self.centroids.weight)
            attention_scores = self.calc_stochastic_attention(torch.matmul(query_layer, key_centroids.transpose(-1, -2)), self.tau_2)
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ElectraModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # stochastic modification
        if not self.hierarchial:
            # TODO:
            # mb move before position embeddings
            attention_probs = self.calc_stochastic_attention(attention_scores, self.tau_1)
        else:
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


def replace_attention(model, ue_args, layer_idx):
    model_config = model.electra.encoder.config
    position_embedding_type = model.electra.encoder.layer[layer_idx].attention.self.position_embedding_type
    new_attention = ElectraSelfAttentionStochastic(model_config,
                                                   position_embedding_type,
                                                   ue_args.hierarchial,
                                                   ue_args.num_centroids,
                                                   ue_args.tau_1, ue_args.tau_2,
                                                   )
    new_attention.post_init(model.electra.encoder.layer[layer_idx].attention.self)
    model.electra.encoder.layer[layer_idx].attention.self = new_attention
    return model


def change_attention_params(model, ue_args, layers="all"):
    if layers == "all":
        for layer_idx, _ in enumerate(model.electra.encoder.layer):
            model.electra.encoder.layer[layer_idx].attention.self.hierarchial = ue_args.hierarchial
            model.electra.encoder.layer[layer_idx].attention.self.num_centroids = ue_args.num_centroids
            model.electra.encoder.layer[layer_idx].attention.self.tau_1 = ue_args.tau_1
            model.electra.encoder.layer[layer_idx].attention.self.tau_2 = ue_args.tau_2
    else:
        layer_idx = -1
        model.electra.encoder.layer[layer_idx].attention.self.hierarchial = ue_args.hierarchial
        model.electra.encoder.layer[layer_idx].attention.self.num_centroids = ue_args.num_centroids
        model.electra.encoder.layer[layer_idx].attention.self.tau_1 = ue_args.tau_1
        model.electra.encoder.layer[layer_idx].attention.self.tau_2 = ue_args.tau_2
    return model


class ElectraClassificationHS(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other, n_labels, hs_labels=None):
        super().__init__()
        
        self.dropout1 = other.dropout
        self.dense = other.dense
        
        if hs_labels is None:
            self.H = other.dense.in_features
            self.hs_labels = torch.randn((n_labels, self.H))

            def HS_cost_func(X):
                Z = X@X.T - 2*torch.eye(n_labels)
                return Z.max(axis=1).values.mean()

            # L-BFGS
            self.hs_labels.requires_grad = True

            #optimizer = torch.optim.LBFGS([self.hs_labels],
            #                                history_size=100,
            #                                max_iter=200,
            #                                line_search_fn="strong_wolfe")
            
            optimizer = torch.optim.SGD([self.hs_labels], lr=2e-1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
            eps = 1e-7
            losses = []
            
            for i in range(5000):
                optimizer.zero_grad()
                objective = HS_cost_func(self.hs_labels)
                objective.backward()
                optimizer.step(lambda: HS_cost_func(self.hs_labels))
                scheduler.step()
                losses.append(HS_cost_func(self.hs_labels).detach().numpy())
                if len(losses) > 2 and (np.abs(losses[-2] - losses[-1]) < eps):
                    break

            self.hs_labels.requires_grad = False
        else:
            self.hs_labels = hs_labels
    
    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout1(x)
        x = self.dense(x)
        self.hs_labels = self.hs_labels.to(x.device)
        x = torch.matmul(x, self.hs_labels.T) * torch.norm(self.hs_labels)
        return x


class BERTClassificationHS(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other, n_labels, hs_labels=None):
        super().__init__()
        if hs_labels is None:
            self.H = other.in_features
            self.hs_labels = torch.randn((n_labels, self.H))

            def HS_cost_func(X):
                Z = X@X.T - 2*torch.eye(n_labels)
                return Z.max(axis=1).values.mean()

            # L-BFGS
            self.hs_labels.requires_grad = True

            optimizer = torch.optim.LBFGS([self.hs_labels],
                                            history_size=10,
                                            max_iter=4,
                                            line_search_fn="strong_wolfe")
            for i in range(5):
                optimizer.zero_grad()
                objective = HS_cost_func(self.hs_labels)
                objective.backward()
                optimizer.step(lambda: HS_cost_func(self.hs_labels))

            self.hs_labels.requires_grad = False
        else:
            self.hs_labels = hs_labels
    
    def forward(self, x, **kwargs):
        self.hs_labels = self.hs_labels.to(x.device)
        x = torch.matmul(x, self.hs_labels.T) * torch.norm(self.hs_labels)
        return x
