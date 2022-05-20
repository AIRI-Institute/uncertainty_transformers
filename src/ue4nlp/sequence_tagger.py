import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# from al4ner.utils_autodataset import convert_to_ner_dataset
from transformers import Trainer
import itertools
from tqdm import trange
import numpy as np
import pickle
import json
import os
from .alpaca_calibrator import ModelWithTempScaling

import logging

log = logging.getLogger("sequence_tagger_auto")


class SequenceTagger:
    def __init__(
        self,
        auto_model,
        bpe_tokenizer,
        max_len=192,
        pred_loader_args={"num_workers": 1},
        pred_batch_size=100,
        training_args=None,
        trainer=None,
        sngp_output=False
    ):
        super().__init__()

        self._auto_model = auto_model
        self._bpe_tokenizer = bpe_tokenizer
        self._pred_loader_args = pred_loader_args
        self._pred_batch_size = pred_batch_size
        self._training_args = training_args
        self._trainer = trainer
        self._named_parameters = auto_model.named_parameters
        self.temperature = 1
        self.sngp_output = sngp_output

    @property
    def _bert_model(self):
        return self._auto_model

    @property
    def model(self):
        return self._auto_model

    def predict(
        self,
        eval_dataset,
        evaluate=False,
        metrics=None,
        calibrate=False,
        apply_softmax=True,
        return_preds=True,
    ):
        if metrics is None:
            metrics = []

        self._auto_model.eval()

        if not self.sngp_output:
            logits, labels, metrics = self._trainer.predict(eval_dataset)
        else:
            logits, stds, labels, metrics = self._trainer.predict(eval_dataset)
        
        if calibrate:
            labels = torch.as_tensor([example["labels"] for example in eval_dataset])

            logits = logits.transpose((0, 2, 1))
            labels = labels.reshape(logits.shape[0], -1)

            calibr = ModelWithTempScaling(self._auto_model)
            calibr.scaling(
                torch.FloatTensor(logits),
                torch.LongTensor(labels),
                lr=1e-3,
                max_iter=50,
            )
            self.temperature = calibr.temperature.detach().numpy()[0]
            self.temperature = np.clip(self.temperature, -10, 10)

        if not (apply_softmax) and not (return_preds):
            # special case for NUQ and Mahalanobis estimators
            return (logits, logits, labels, metrics)
        probs = F.softmax(
            torch.tensor(np.true_divide(logits, self.temperature)), dim=2
        ).numpy()
        preds = np.argmax(probs, axis=-1)

        log.info(metrics)
        
        if self.sngp_output:
            return preds, probs, logits, stds, labels, metrics

        return preds, probs, labels
