import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from tqdm import tqdm
import hydra
from pathlib import Path
import json
import numpy as np

from sklearn.metrics import accuracy_score

from ue4nlp.dropout_mc import DropoutMC, convert_to_mc_dropout, activate_mc_dropout
from ue4nlp.dropout_dpp import DropoutDPP

# from alpaca.uncertainty_estimator import build_estimator
from estimators_debug import BaldMasked
from alpaca.uncertainty_estimator.masks import build_mask

import logging

log = logging.getLogger(__name__)


def convert_dropouts(ue_args, model):
    if ue_args.mc_type == "MC":
        if ue_args.dropout_subs == "last":
            model.set_last_dropout(
                DropoutMC(p=model.get_last_dropout().p, activate=False)
            )
        elif ue_args.dropout_subs == "all":
            convert_to_mc_dropout(model, {"Dropout": DropoutMC})
        else:
            raise ValueError(f"Wrong ue args {ue_args.dropout_subs}")

    elif ue_args.mc_type == "DPP":

        def dropout_ctor(p, activate):
            return DropoutDPP(
                p=p,
                activate=activate,
                max_n=ue_args.dropout.max_n,
                max_frac=ue_args.dropout.max_frac,
                mask_name=ue_args.dropout.mask_name,
            )

        model.set_last_dropout(dropout_ctor(model.get_last_dropout().p, activate=False))

    else:
        raise ValueError(f"Wrong dropout type: {ue_args.mc_type}")


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = x
        return output

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.1)
        # return optimizer, scheduler

    def get_last_dropout(self):
        return self.dropout2

    def set_last_dropout(self, dropout):
        self.dropout2 = dropout


class SimpleConv(pl.LightningModule):
    def __init__(self, num_classes=10, activation=None):
        if activation is None:
            self.activation = F.leaky_relu
        else:
            self.activation = activation
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.linear_size = 12 * 12 * 32
        self.fc1 = nn.Linear(self.linear_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, dropout_rate=0.0, dropout_mask=None):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.linear_size)
        x = self.activation(self.fc1(x))
        x = self._dropout(x, dropout_mask, dropout_rate, 0)
        x = self.fc2(x)
        return x

    def _dropout(self, x, dropout_mask, dropout_rate, layer_num):
        if dropout_mask is None:
            x = self.dropout(x)
        else:
            x = x * dropout_mask(x, dropout_rate, layer_num)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def get_last_dropout(self):
        return self.dropout

    def set_last_dropout(self, dropout):
        self.dropout = dropout


class StrongConv(pl.LightningModule):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        base = 16
        self.conv = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.CELU(),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.CELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            nn.Conv2d(base, 2 * base, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * base),
            nn.CELU(),
            nn.Conv2d(2 * base, 2 * base, 3, padding=1, bias=False),
            nn.CELU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2 * base, 4 * base, 3, padding=1, bias=False),
            nn.BatchNorm2d(4 * base),
            nn.CELU(),
            nn.Conv2d(4 * base, 4 * base, 3, padding=1, bias=False),
            nn.CELU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2, 2),
        )
        self.linear_size = 8 * 8 * base
        self.linear = nn.Sequential(
            nn.Linear(self.linear_size, 8 * base),
            nn.CELU(),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(8 * base, 10)

    def forward(self, x, dropout_rate=0.5, dropout_mask=None):
        x = self.conv(x)
        x = x.reshape(-1, self.linear_size)
        x = self.linear(x)
        if dropout_mask is None:
            x = self.dropout(x)
        else:
            x = x * dropout_mask(x, dropout_rate, 0)
        return self.fc(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def get_last_dropout(self):
        return self.dropout

    def set_last_dropout(self, dropout):
        self.dropout = dropout


def predict(mnist_model, dataset):
    mnist_model.eval()
    with torch.no_grad():
        logits = []
        for data_batch, target_batch in dataset:
            logits.append(mnist_model(data_batch.cuda()))

        probas = torch.softmax(torch.cat(logits, dim=0), dim=1)
        preds = probas.argmax(dim=1)

        return probas.cpu(), preds.cpu()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# def double_transform(img):
#     return transforms.ToTensor()(img).double()


@hydra.main(config_path=os.environ["HYDRA_CONFIG_PATH"])
def main(configs):
    auto_generated_dir = os.getcwd()
    log.info(f"Work dir: {auto_generated_dir}")
    os.chdir(hydra.utils.get_original_cwd())

    set_seed(configs.seed)

    train_dataset = MNIST(
        configs.data.data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    if configs.data.subsample_perc > 0:
        dataset_len = int(configs.data.subsample_perc * len(train_dataset))
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [dataset_len, len(train_dataset) - dataset_len]
        )

    log.info(f"Dataset len: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = MNIST(
        configs.data.data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_loader = DataLoader(test_dataset, batch_size=10000)

    # mnist_model = MNISTModel().cuda()
    mnist_model = SimpleConv().cuda()
    mnist_model.train()
    trainer = pl.Trainer(
        progress_bar_refresh_rate=5,
        max_epochs=10,
        auto_select_gpus=True,
        gpus=1,
        early_stop_callback=True,
    )
    trainer.fit(mnist_model, train_loader)

    eval_results = {
        "sampled_probabilities": [],
        "sampled_answers": [],
        "probabilities": [],
        "answers": [],
        "true_labels": [l for b, l in test_dataset],
    }

    mnist_model = mnist_model.cuda()
    probs, answers = predict(mnist_model, test_loader)
    acc_score = accuracy_score(eval_results["true_labels"], answers)
    log.info(f"#####Accuracy score: {acc_score}")

    eval_results["probabilities"] = probs.tolist()
    eval_results["answers"] = answers.tolist()

    log.info("*********************Perform stochastic inference********************")

    convert_dropouts(configs.ue, mnist_model)

    activate_mc_dropout(mnist_model, activate=True, random=configs.ue.inference_prob)

    if configs.ue.mc_type == "DPP":
        log.info("*****************Dry run********************")
        dpp_dropout = mnist_model.get_last_dropout()
        dpp_dropout.mask.freeze(dry_run=True)
        predict(mnist_model, test_loader)
        dpp_dropout.mask.unfreeze(dry_run=True)

    log.info("Predict")
    for _ in tqdm(range(configs.ue.committee_size)):
        probs, preds = predict(mnist_model, test_loader)
        eval_results["sampled_probabilities"].append(probs.tolist())
        eval_results["sampled_answers"].append(preds.tolist())

    activate_mc_dropout(mnist_model, activate=False)

    with open(Path(auto_generated_dir) / "dev_inference.json", "w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
