import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime

from transformers import ElectraForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


def calc_gradient_penalty(x, y_pred_sum):
    gradients = torch.autograd.grad(
        outputs=y_pred_sum,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred_sum),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.flatten(start_dim=1)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty


class ElectraDUQModel(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        classifier,
        output_dir,
        length_scale,
        embedding_size,
        gamma,
        batch_size,
        l_gradient_penalty,
    ):
        super().__init__()
        feat_size = classifier.dense.in_features  # feat. size from encoder
        num_classes = classifier.out_proj.out_features
        self.class_type = "DUQ"
        self.mode = "train"
        self.output_dir = output_dir
        self.l_gradient_penalty = l_gradient_penalty
        self.writer = SummaryWriter(os.path.join(self.output_dir, "tboard"))
        learnable_length_scale = False
        self.length_scale = length_scale
        if embedding_size is None:
            embedding_size = feat_size
        self.embedding_size = embedding_size
        self.gamma = gamma

        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, feat_size), 0.05)
        )

        self.register_buffer(
            "N", torch.ones(num_classes) * (batch_size / num_classes)
        )  # 8 = batch_size / num_classes
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

    def update_embeddings(self, features, y):
        x = features[:, 0, :]
        z = self.last_layer(x)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1.0 - self.gamma) * y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1.0 - self.gamma) * features_sum

    def last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(self.N * 2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        z = self.last_layer(x)
        y_pred = self.output_layer(z)
        # print(y_pred[:, 0])

        return y_pred


class ElectraForSequenceClassificationDUQ(ElectraForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.idx = 0

    def make_duq(self, output_dir, batch_size, duq_params):
        duq_params = {
            "classifier": self.classifier,
            "length_scale": duq_params["length_scale"],
            "l_gradient_penalty": duq_params["l_gradient_penalty"],
            "embedding_size": duq_params["embedding_size"],
            "gamma": duq_params["gamma"],
            "batch_size": batch_size,
            "output_dir": output_dir,
        }
        self.classifier = ElectraDUQModel(**duq_params)

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
        inputs_embeds = self.electra.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
        )

        discriminator_hidden_states = self.electra(
            input_ids=None,  # input_ids,
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

        writer = self.classifier.writer
        l_gradient_penalty = self.classifier.l_gradient_penalty
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif hasattr(self.classifier, "class_type"):  # work with DUQ
                y = (
                    F.one_hot(labels, num_classes=self.num_labels)
                    .float()
                    .to(logits.device)
                )
                # logits here are not really logits, they are exp (-distances to) centroids
                loss = F.binary_cross_entropy(logits, y)
                if self.training:
                    # Calculate gradient penalty
                    gradients = torch.autograd.grad(
                        outputs=logits.sum(1),
                        inputs=inputs_embeds,
                        grad_outputs=torch.ones_like(logits.sum(1)),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    gradients = gradients.flatten(start_dim=1)
                    # L2 norm
                    grad_norm = gradients.norm(2, dim=1)
                    # Two sided penalty
                    gradient_penalty = ((grad_norm - 1) ** 2).mean()

                    grad_pen = l_gradient_penalty * gradient_penalty

                    with torch.no_grad():
                        cossim = torch.nn.functional.cosine_similarity(
                            self.classifier.m.t()[:, :, None],
                            self.classifier.m[None, :, :],
                        )
                        preds = torch.argmax(logits, dim=1)
                        std = torch.std(sequence_output[:, 0, :], dim=-1).mean()
                    writer.add_scalar("bce", loss.item(), self.idx)
                    writer.add_scalar("grad_pen", grad_pen.item(), self.idx)
                    writer.add_scalar(
                        "acc",
                        (preds == labels).type(torch.float32).mean().item(),
                        self.idx,
                    )
                    writer.add_scalar("mean_cossim", cossim.mean().item(), self.idx)
                    writer.add_scalar("features_std", std.item(), self.idx)
                    writer.flush()

                    self.idx += 1
                    # loss = loss + grad_pen
                    with torch.no_grad():
                        self.eval()
                        discriminator_hidden_states_eval = self.electra(
                            # input_ids=input_ids,
                            input_ids=None,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict,
                        )

                        sequence_output_eval = discriminator_hidden_states_eval[0]
                        self.classifier.update_embeddings(sequence_output_eval, y)
                        self.train()
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
