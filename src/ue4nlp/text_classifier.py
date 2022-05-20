import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from transformers import Trainer
import itertools
from tqdm import trange
import numpy as np
import pickle
import json
import os

from . import alpaca_calibrator as calibrator

import logging

log = logging.getLogger("text_classifier")


class TextClassifier:
    def __init__(
        self,
        auto_model,
        bpe_tokenizer,
        max_len=192,
        pred_loader_args={"num_workers": 1},
        pred_batch_size=100,
        training_args=None,
        trainer=None,
        use_paper_version=False,
    ):
        super().__init__()

        self._auto_model = auto_model
        self._bpe_tokenizer = bpe_tokenizer
        self._pred_loader_args = pred_loader_args
        self._pred_batch_size = pred_batch_size
        self._training_args = training_args
        self._trainer = trainer
        self._named_parameters = auto_model.named_parameters
        self.temperature = 1.0
        self._max_len = max_len
        self.use_paper_version = use_paper_version

    @property
    def _bert_model(self):
        return self._auto_model

    @property
    def model(self):
        return self._auto_model

    @property
    def tokenizer(self):
        return self._bpe_tokenizer

    def predict(
        self,
        eval_dataset,
        calibrate=False,
        apply_softmax=True,
        return_preds=True,
    ):
        self._auto_model.eval()

        res = self._trainer.predict(eval_dataset)
        logits = res[0]
        if isinstance(logits, tuple):
            logits = logits[0]
        if calibrate:
            labels = [example["label"] for example in eval_dataset]
            calibr = calibrator.ModelWithTempScaling(self._auto_model)
            if self.use_paper_version:
                max_iter = 50
                calibrate_lower = -10
            else:
                max_iter = 100
                calibrate_lower = 0.1
            calibr.scaling(
                torch.FloatTensor(logits),
                torch.LongTensor(labels),
                lr=1e-3,  # TODO:
                max_iter=max_iter,  # TODO:
            )
            self.temperature = calibr.temperature.detach().numpy()[0]
            self.temperature = np.clip(self.temperature, calibrate_lower, 10)

        logits = np.true_divide(logits, self.temperature)

        if apply_softmax:
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        else:
            probs = logits

        if not return_preds:
            return [probs] + list(res)

        preds = np.argmax(probs, axis=1)

        return [preds, probs] + list(res)
