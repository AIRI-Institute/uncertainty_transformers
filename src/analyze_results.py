import os
from pathlib import Path
import json as stable_json # use in case then orjson failed
import orjson as json  # faster
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import yaml
from yaml import Loader as Loader
import re

from sklearn.metrics import (
    accuracy_score,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
from ue4nlp.ue_scores import *

import logging

log = logging.getLogger()

default_methods = {
    "bald": bald,
    "sampled_max_prob": sampled_max_prob,
    "variance": probability_variance,
}


def unpad_preds(probs, sampled_probs, preds, labels):
    true_sampled_probs = [
        [p.tolist() for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(sampled_probs.transpose(1, 2, 3, 0), labels[:, :])
    ]
    true_probs = [
        [p.tolist() for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(probs, labels[:, :])
    ]
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels[:, :])
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels[:, :])
    ]

    return true_sampled_probs, true_probs, true_predictions, true_labels


def get_score_ratio_seq(sorted_indexes, answers, true_answers, ratio):
    last_index = int(len(sorted_indexes) * ratio)
    sel_indexes = sorted_indexes[:last_index]
    unsel_indexes = sorted_indexes[last_index:]

    sel_answers = []
    for ind in sel_indexes:
        sel_answers.append(true_answers[ind])
    for ind in unsel_indexes:
        sel_answers.append(answers[ind])

    sel_true_answers = []
    for ind in sel_indexes:
        sel_true_answers.append(true_answers[ind])
    for ind in unsel_indexes:
        sel_true_answers.append(true_answers[ind])

    score = sum([1.0 * (l == p) for l, p in zip(sel_answers, sel_true_answers)]) / len(
        sel_answers
    )
    return score


def get_score_ratio(sorted_indexes, answers, true_answers, ratio, metric=accuracy_score, drop=False):
    last_index = int(len(sorted_indexes) * ratio)
    sel_indexes = sorted_indexes[:last_index]
    unsel_indexes = sorted_indexes[last_index:]

    if drop:
        sel_answers = answers[unsel_indexes].tolist()
        sel_true_answers = true_answers[unsel_indexes].tolist()
    else:
        sel_answers = (
            true_answers[sel_indexes].tolist() + answers[unsel_indexes].tolist()
        )
        sel_true_answers = (
            true_answers[sel_indexes].tolist() + true_answers[unsel_indexes].tolist()
        )
    score = metric(sel_true_answers, sel_answers)
    return score

def is_ue_score(name):
    return ("mahalanobis" in name or "nuq" in name or "mixup" in name or "ddu" in name or "disc" in name)

def calc_rejection_curve_aucs(
    probabilities, labels, sampled_probabilities, model_answers, methods
):
    ratio_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    predictions = np.argmax(probabilities, axis=-1)
    errors = (labels != predictions).astype("uint8")

    model_ues = 1 - np.max(probabilities, axis=1)
    sorted_indexes_model = np.argsort(-model_ues)
    results = {}
    model_scores = [
        get_score_ratio(sorted_indexes_model, model_answers, labels, ratio)
        for ratio in ratio_list
    ]
    results["max_prob"] = auc(ratio_list, model_scores)

    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)
        ensemble_answers = np.asarray(sampled_probabilities).mean(1).argmax(-1)
        sorted_indexes_ensemble = np.argsort(-ue_scores)
        if is_ue_score(name):
            # because for this case we have ue scores in sampled_probabilities
            ensemble_answers = predictions
        ens_scores = [
            get_score_ratio(sorted_indexes_ensemble, ensemble_answers, labels, ratio)
            for ratio in ratio_list
        ]
        results[name] = auc(ratio_list, ens_scores)
    return results


def calc_rejection_curve_auc_seq(probs, labels, sampled_probs, model_answers, methods, avg_type='sum'):
    sampled_probs, probs, predictions, labels = unpad_preds(
        probs, sampled_probs, np.argmax(probs, axis=-1), labels
    )

    if methods is None:
        methods = default_methods

    ratio_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    errors = [1.0 * (l != p) for l, p in zip(labels, predictions)]

    n_examples = len(errors)
    ue_scores_max = np.zeros(n_examples)
    for i in range(n_examples):
        sent = probs[i]
        true_probs_max = np.asarray([np.max(proba) for proba in sent])
        ue_scores_max[i] = np.mean(1 - true_probs_max)

    sorted_indexes_model = np.argsort(-ue_scores_max)

    results = {}
    model_scores = [
        get_score_ratio_seq(sorted_indexes_model, predictions, labels, ratio)
        for ratio in ratio_list
    ]
    results["max_prob"] = auc(ratio_list, model_scores)

    for name, method_function in methods.items():
        ensemble_answers = [
            np.asarray(p).mean(-1).argmax(-1).tolist() for p in sampled_probs
        ]
        
        if is_ue_score(name):
            # because for this case we have ue scores in sampled_probabilities
            avg_type = 'max'
            ensemble_answers = predictions
            
        ue_scores = seq_ue(sampled_probs, method_function, avg_type=avg_type)
        sorted_indexes_ensemble = np.argsort(-ue_scores)

        ens_scores = [
            get_score_ratio_seq(
                sorted_indexes_ensemble, ensemble_answers, labels, ratio
            )
            for ratio in ratio_list
        ]
        results[name] = auc(ratio_list, ens_scores)

    return results


def calc_roc_aucs_seq(labels, probs, sampled_probs, methods=None, avg_type='sum'):

    sampled_probs, probs, predictions, labels = unpad_preds(
        probs, sampled_probs, np.argmax(probs, axis=-1), labels
    )

    if methods is None:
        methods = default_methods

    errors = [1.0 * (l != p) for l, p in zip(labels, predictions)]
    results = {}
    for name, method_function in methods.items():
        if is_ue_score(name):
            avg_type = 'max'
        ue_scores = seq_ue(sampled_probs, method_function, avg_type=avg_type)
        results[name] = roc_auc_score(errors, ue_scores)

    n_examples = len(errors)
    ue_scores_max = np.zeros(n_examples)
    for i in range(n_examples):
        sent = probs[i]
        true_probs_max = np.asarray([np.max(proba) for proba in sent])
        ue_scores_max[i] = np.mean(1 - true_probs_max)

    results["max_prob"] = roc_auc_score(errors, ue_scores_max)
    return results


def calc_roc_aucs(
    probabilities, labels, sampled_probabilities, methods, oos=False, top3=False
):
    predictions = np.argmax(probabilities, axis=-1)
    if oos:
        if len(np.unique(labels)) > 40:
            #CLINC use class №42 as OOD
            errors = (labels == 42).astype("uint8")
        else:
            #SNIPS and ROSTD case
            errors = (labels == np.max(labels)).astype("uint8")
    elif top3:
        top3 = np.argsort(probabilities, axis=-1)[:, -3:]
        errors = np.array(
            [(l not in top3[i]) * 1 for i, l in enumerate(labels)]
        ).astype("uint8")
    else:
        #misclassification case
        errors = (labels != predictions).astype("uint8")

    results = {}
    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)
        results[name] = roc_auc_score(errors, ue_scores)

    max_prob = 1.0 - np.max(probabilities, axis=-1)
    results["max_prob"] = roc_auc_score(errors, max_prob)
    return results

def rcc_auc(conf, risk, return_points=False):
    # risk-coverage curve's area under curve
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=True)

    cumulative_risk = [cr_pair[0][1]]
    for i in range(1, n):
        cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])

    points_x = []
    points_y = []

    auc = 0
    for k in range(n):
        auc += cumulative_risk[k] / (1 + k)
        points_x.append((1 + k) / n)  # coverage
        points_y.append(cumulative_risk[k] / (1 + k))  # current avg. risk

    if return_points:
        return auc, points_x, points_y
    else:
        return auc


def calc_rcc_aucs(probabilities, labels, sampled_probabilities, methods):
    predictions = np.argmax(probabilities, axis=-1)

    risk_binary = (predictions != labels).astype(int)

    conf = np.max(probabilities, axis=1)
    results = {}

    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)
        results[name] = rcc_auc(-ue_scores, risk_binary)

    results["max_prob"] = rcc_auc(conf, risk_binary)
    return results


def rpp(conf, risk):
    # reverse pair proportion
    # for now only works when risk is binary
    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=False)

    pos_count, rp_count = 0, 0
    for i in range(n):
        if cr_pair[i][1] == 0:  # risk==0
            pos_count += 1
        else:
            rp_count += pos_count

    return rp_count / (n ** 2)


def calc_rpp(probabilities, labels, sampled_probabilities, methods):
    predictions = np.argmax(probabilities, axis=-1)

    risk_binary = (predictions != labels).astype(int)

    conf = np.max(probabilities, axis=1)
    results = {}

    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)
        results[name] = rpp(-ue_scores, risk_binary)

    results["max_prob"] = rpp(conf, risk_binary)
    return results


def calc_pr_aucs(
    answers, probabilities, eval_labels, sampled_probabilities, methods, oos
):
    if not oos:
        labels = (eval_labels != answers).astype("uint8")
    elif len(np.unique(eval_labels)) > 40:
        labels = (eval_labels == 42).astype("uint8")
    else:
        labels = (eval_labels == np.max(eval_labels)).astype("uint8")

    results = {}
    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)
        results[name] = average_precision_score(labels, ue_scores)
    max_prob = 1.0 - np.max(probabilities, axis=-1)
    results["max_prob"] = average_precision_score(labels, max_prob)
    return results

def calc_precision(
    answers, probabilities, eval_labels, sampled_probabilities, methods, oos
):
    if not oos:
        labels = (eval_labels != answers).astype("uint8")
    elif len(np.unique(eval_labels)) > 40:
        labels = (eval_labels == 42).astype("uint8")
    else:
        labels = (eval_labels == np.max(eval_labels)).astype("uint8")

    results = {}
    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)
        precision, recall, thresholds = precision_recall_curve(labels, ue_scores)
        f1_score = 2*precision*recall/(precision+recall+1e-7)
        results[name] = precision[np.argmax(f1_score)]
    max_prob = 1.0 - np.max(probabilities, axis=-1)
    precision, recall, thresholds = precision_recall_curve(labels, ue_scores)
    f1_score = 2*precision*recall/(precision+recall+1e-7)
    results["max_prob"] = precision[np.argmax(f1_score)]
    return results

def calc_recall(
    answers, probabilities, eval_labels, sampled_probabilities, methods, oos
):
    if not oos:
        labels = (eval_labels != answers).astype("uint8")
    elif len(np.unique(eval_labels)) > 40:
        labels = (eval_labels == 42).astype("uint8")
    else:
        labels = (eval_labels == np.max(eval_labels)).astype("uint8")

    results = {}
    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)
        precision, recall, thresholds = precision_recall_curve(labels, ue_scores)
        f1_score = 2*precision*recall/(precision+recall+1e-7)
        results[name] = recall[np.argmax(f1_score)]
    max_prob = 1.0 - np.max(probabilities, axis=-1)
    precision, recall, thresholds = precision_recall_curve(labels, ue_scores)
    f1_score = 2*precision*recall/(precision+recall+1e-7)
    results["max_prob"] = recall[np.argmax(f1_score)]
    return results

def calc_f1_score(
    answers, probabilities, eval_labels, sampled_probabilities, methods, oos
):
    if not oos:
        labels = (eval_labels != answers).astype("uint8")
    elif len(np.unique(eval_labels)) > 40:
        labels = (eval_labels == 42).astype("uint8")
    else:
        labels = (eval_labels == np.max(eval_labels)).astype("uint8")

    results = {}
    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)
        precision, recall, thresholds = precision_recall_curve(labels, ue_scores)
        precision, recall = np.array(precision), np.array(recall)
        f1_score = 2*precision*recall/(precision+recall+1e-7)
        results[name] = np.max(f1_score)
    max_prob = 1.0 - np.max(probabilities, axis=-1)
    precision, recall, thresholds = precision_recall_curve(labels, max_prob)
    precision, recall = np.array(precision), np.array(recall)
    f1_score = 2*precision*recall/(precision+recall+1e-7)
    results["max_prob"] = np.max(f1_score)
    return results


def calc_rcc_aucs_seq(
    probabilities, labels, sampled_probabilities, predictions, methods, avg_type='sum'
):
    risk_binary = [1.0 * (l != p) for l, p in zip(labels, predictions)]

    results = {}

    # all this methods are experimental, for now look only on results['rcc_auc']
    for name, method_function in methods.items():
        if is_ue_score(name):
            avg_type = 'max'
        ue_scores = seq_ue(sampled_probabilities, method_function, avg_type=avg_type)
        results[name] = rcc_auc(-ue_scores, risk_binary)

    n_examples = len(risk_binary)
    ue_scores_max = np.zeros(n_examples)
    for i in range(n_examples):
        sent = probabilities[i]
        true_probs_max = np.asarray([np.max(proba) for proba in sent])
        ue_scores_max[i] = np.mean(true_probs_max)
    results["max_prob"] = rcc_auc(ue_scores_max, risk_binary)
    return results


def calc_rpp_seq(probabilities, labels, sampled_probabilities, predictions, methods, avg_type='sum'):
    risk_binary = [1.0 * (l != p) for l, p in zip(labels, predictions)]

    results = {}

    # all this methods are experimental, for now look only on results['rcc_auc']
    for name, method_function in methods.items():
        if is_ue_score(name):
            avg_type = 'max'
        ue_scores = seq_ue(sampled_probabilities, method_function, avg_type=avg_type)
        results[name] = rpp(-ue_scores, risk_binary)

    n_examples = len(risk_binary)
    ue_scores_max = np.zeros(n_examples)
    for i in range(n_examples):
        sent = probabilities[i]
        true_probs_max = np.asarray([np.max(proba) for proba in sent])
        ue_scores_max[i] = np.mean(true_probs_max)
    results["max_prob"] = rpp(ue_scores_max, risk_binary)
    return results


def calc_pr_aucs_seq(
    answers, probabilities, labels, sampled_probabilities, predictions, methods, avg_type='sum'
):
    errors = [1.0 * (l != p) for l, p in zip(labels, predictions)]
    results = {}

    for name, method_function in methods.items():
        if is_ue_score(name):
            avg_type = 'max'
        ue_scores = seq_ue(sampled_probabilities, method_function, avg_type=avg_type)
        results[name] = average_precision_score(errors, ue_scores)

    n_examples = len(errors)
    ue_scores_max = np.zeros(n_examples)
    for i in range(n_examples):
        sent = probabilities[i]
        true_probs_max = np.asarray([np.max(proba) for proba in sent])
        ue_scores_max[i] = np.mean(1 - true_probs_max)

    results["max_prob"] = average_precision_score(errors, ue_scores_max)
    return results

def from_model_outputs_calc_recall(model_outputs, methods, oos=False):
    if "sampled_probabilities" in model_outputs.keys():
        res = calc_recall(
            np.asarray(model_outputs["answers"]),
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            methods=methods,
            oos=oos,
        )
    else:
        raise ValueError(f"Error keys {model_outputs.keys()}")
    return res

def from_model_outputs_calc_precision(model_outputs, methods, oos=False):
    if "sampled_probabilities" in model_outputs.keys():
        res = calc_precision(
            np.asarray(model_outputs["answers"]),
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            methods=methods,
            oos=oos,
        )
    else:
        raise ValueError(f"Error keys {model_outputs.keys()}")
    return res

def from_model_outputs_calc_pr_auc(model_outputs, methods, oos=False):
    if "sampled_probabilities" in model_outputs.keys():
        res = calc_pr_aucs(
            np.asarray(model_outputs["answers"]),
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            methods=methods,
            oos=oos,
        )
    else:
        raise ValueError(f"Error keys {model_outputs.keys()}")
    return res

def from_model_outputs_calc_f1_score(model_outputs, methods, oos=False):
    if "sampled_probabilities" in model_outputs.keys():
        res = calc_f1_score(
            np.asarray(model_outputs["answers"]),
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            methods=methods,
            oos=oos,
        )
    else:
        raise ValueError(f"Error keys {model_outputs.keys()}")
    return res

def from_model_outputs_calc_rpp(model_outputs, methods):
    if "sampled_probabilities" in model_outputs.keys():
        res = calc_rpp(
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            methods=methods,
        )
    else:
        raise ValueError(f"Error keys {model_outputs.keys()}")
    return res

def from_model_outputs_calc_rcc_auc(model_outputs, methods=None):
    used_methods = methods if methods is not None else default_methods
    if "sampled_probabilities" in model_outputs.keys():
        res = calc_rcc_aucs(
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            methods=methods,
        )
    else:
        raise ValueError(f"Error keys {model_outputs.keys()}")
    return res

def from_model_outputs_calc_arc_auc(model_outputs, methods):
    if "sampled_probabilities" in model_outputs.keys() and 'stds' not in model_outputs.keys():
        res = calc_rejection_curve_aucs(
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            np.asarray(model_outputs["answers"]),
            methods=methods,
        )
    else:
        raise ValueError(f"Error keys {model_outputs.keys()}")
    return res

def from_model_outputs_calc_roc_auc(model_outputs, methods):
    if "sampled_probabilities" in model_outputs.keys():
        res = calc_roc_aucs(
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            methods=methods,
        )
    else:
        raise ValueError(f"Error keys {model_outputs.keys()}")
    return res


def from_model_outputs_calc_rcc_auc_ner(model_outputs, methods, level="token", avg_type='sum'):
    probs = np.asarray(model_outputs["probabilities"])
    probs_toks = probs.reshape(-1, probs.shape[-1])

    sampled_probs = np.asarray(model_outputs["sampled_probabilities"])
    sampled_probs_toks = sampled_probs.reshape(
        sampled_probs.shape[0], sampled_probs.shape[1] * sampled_probs.shape[2], -1
    )

    labels = np.asarray(model_outputs["true_labels"])
    labels_toks = labels.reshape(-1)

    use_idx = labels_toks != -100

    if level == "token":
        res = calc_rcc_aucs(
            probs_toks[use_idx],
            labels_toks[use_idx],
            sampled_probs_toks[:, use_idx].transpose(1, 0, 2),
            methods=methods,
        )
    else:
        # sequence level
        sampled_probs, probs, predictions, labels = unpad_preds(
            probs, sampled_probs, np.argmax(probs, axis=-1), labels
        )
        res = calc_rcc_aucs_seq(
            probs,
            labels,
            sampled_probs,
            predictions,
            methods=methods,
            avg_type=avg_type,
        )
    return res


def from_model_outputs_calc_pr_auc_ner(model_outputs, methods, level="token"):
    probs = np.asarray(model_outputs["probabilities"])
    probs_toks = probs.reshape(-1, probs.shape[-1])

    sampled_probs = np.asarray(model_outputs["sampled_probabilities"])
    sampled_probs_toks = sampled_probs.reshape(
        sampled_probs.shape[0], sampled_probs.shape[1] * sampled_probs.shape[2], -1
    )

    labels = np.asarray(model_outputs["true_labels"])
    labels_toks = labels.reshape(-1)

    use_idx = labels_toks != -100

    if level == "token":
        res = calc_pr_aucs(
            np.asarray(model_outputs["answers"]).reshape(-1)[use_idx],
            probs_toks[use_idx],
            labels_toks[use_idx],
            sampled_probs_toks[:, use_idx].transpose(1, 0, 2),
            methods=methods,
            oos=False,
        )
    else:
        # sequence level
        sampled_probs, probs, predictions, labels = unpad_preds(
            probs, sampled_probs, np.argmax(probs, axis=-1), labels
        )
        res = calc_pr_aucs_seq(
            np.asarray(model_outputs["answers"]),
            probs,
            labels,
            sampled_probs,
            predictions,
            methods=methods,
        )
    return res


def from_model_outputs_calc_rpp_ner(model_outputs, methods, level="token", avg_type='sum'):
    probs = np.asarray(model_outputs["probabilities"])
    probs_toks = probs.reshape(-1, probs.shape[-1])

    sampled_probs = np.asarray(model_outputs["sampled_probabilities"])
    sampled_probs_toks = sampled_probs.reshape(
        sampled_probs.shape[0], sampled_probs.shape[1] * sampled_probs.shape[2], -1
    )

    labels = np.asarray(model_outputs["true_labels"])
    labels_toks = labels.reshape(-1)

    use_idx = labels_toks != -100

    if level == "token":
        res = calc_rpp(
            probs_toks[use_idx],
            labels_toks[use_idx],
            sampled_probs_toks[:, use_idx].transpose(1, 0, 2),
            methods=methods,
        )
    else:
        # sequence level
        sampled_probs, probs, predictions, labels = unpad_preds(
            probs, sampled_probs, np.argmax(probs, axis=-1), labels
        )
        res = calc_rpp_seq(
            probs,
            labels,
            sampled_probs,
            predictions,
            methods=methods,
            avg_type=avg_type
        )
    return res


def extract_time(time_dir):
    try:
        with open(Path(time_dir) / "dev_inference.json", "r") as f:
            model_outputs = json.loads(f.read())
    except:
        with open(Path(time_dir) / "dev_inference.json", "r") as f:
            model_outputs = stable_json.loads(f.read())
    if 'ue_time' in model_outputs.keys():
        if 'ensemble' in str(time_dir):
            return {'ue_time': model_outputs['ue_time'] + model_outputs['eval_time']}
        else:
            return {'ue_time': model_outputs['ue_time']}
    else:
        return {'ue_time': 0}


def extract_result(time_dir, methods, metric="roc-auc", oos=False):
    try:
        with open(Path(time_dir) / "dev_inference.json", "r") as f:
            model_outputs = json.loads(f.read())
    except:
        with open(Path(time_dir) / "dev_inference.json", "r") as f:
            model_outputs = stable_json.loads(f.read())
    if "mahalanobis_distance" in model_outputs.keys():
        sampled_probs = np.expand_dims(
            np.asarray(model_outputs["mahalanobis_distance"]), axis=(0, -1)
        )
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
        if "sampled_mahalanobis_distance" in model_outputs.keys():
            sampled_maha = np.expand_dims(
                np.asarray(model_outputs["sampled_mahalanobis_distance"]), axis=(-1)
            )
            sampled_maha = sampled_maha.reshape(
                sampled_maha.shape[0], sampled_probs.shape[1], sampled_probs.shape[2]
            )
            sampled_probs = np.concatenate([sampled_probs, sampled_maha], axis=0)
            model_outputs["sampled_probabilities"] = sampled_probs.tolist()
        elif "mahalanobis_distance_relative" in model_outputs.keys() and not("mahalanobis_distance" in model_outputs.keys()):
            relative_maha = np.expand_dims(
                np.asarray(model_outputs["mahalanobis_distance_relative"]), axis=(0, -1)
            )
            marginal_maha = np.expand_dims(
                np.asarray(model_outputs["mahalanobis_distance_marginal"]), axis=(0, -1)
            )
            sampled_probs = np.stack(
                [sampled_probs, relative_maha, marginal_maha],
                axis=0,
            ).squeeze(axis=1).transpose(1, 0, 2)
            model_outputs["sampled_probabilities"] = sampled_probs.tolist()

    if "uncertainty_score" in model_outputs.keys():
        sampled_probs = np.expand_dims(
            np.asarray(model_outputs["uncertainty_score"]), axis=(0, -1)
        )
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
    elif "ddu_scores" in model_outputs.keys():
        sampled_probs = np.expand_dims(
            np.asarray(model_outputs["ddu_scores"]), axis=(0, -1)
        )
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
        
    elif "stds" in model_outputs.keys():
        sampled_probs = np.expand_dims(
            np.asarray(model_outputs["stds"]).mean(axis=1), axis=(0, -1)
        )
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
    elif "disc_md" in model_outputs.keys():
        sampled_probs_disc_md = np.expand_dims(
            np.asarray(model_outputs["disc_md"]), axis=(-1)
        )
        sampled_probs_nondisc_md = np.expand_dims(
            np.asarray(model_outputs["nondisc_md"]), axis=(-1)
        )
        sampled_probs_total = sampled_probs_disc_md + sampled_probs_nondisc_md
        sampled_probs = np.stack(
            [sampled_probs_disc_md, sampled_probs_nondisc_md, sampled_probs_total],
            axis=0,
        ).transpose(1, 0, 2)
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
    elif "aleatoric" in model_outputs.keys():
        # nuq case - here we have to place all types of uncertainty in sampled_probs
        sampled_probs_aleatoric = np.expand_dims(
            np.asarray(model_outputs["aleatoric"]), axis=(-1)
        )
        sampled_probs_epistemic = np.expand_dims(
            np.asarray(model_outputs["epistemic"]), axis=(-1)
        )
        sampled_probs_total = np.expand_dims(
            np.asarray(model_outputs["total"]), axis=(-1)
        )
        sampled_probs = np.stack(
            [sampled_probs_aleatoric, sampled_probs_epistemic, sampled_probs_total],
            axis=0,
        ).transpose(1, 0, 2)
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
    if metric == "rejection-curve-auc":
        return calc_rejection_curve_aucs(
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            np.asarray(model_outputs["answers"]),
            methods=methods,
        )
    elif metric == "roc-auc":
        return calc_roc_aucs(
            np.asarray(model_outputs["probabilities"]),
            np.asarray(model_outputs["true_labels"]),
            np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
            methods=methods,
            oos=oos,
        )

    elif callable(metric):
        try:
            return metric(model_outputs, methods=methods, oos=oos)
        except:
            return metric(model_outputs, methods=methods)
    else:
        raise ValueError(f"Error metric {metric}")


def extract_result_ner(time_dir, methods, metric="roc-auc", level="token", avg_type='sum'):
    model_outputs, probs, probs_toks, sampled_probs, sampled_probs_toks, labels, labels_toks, use_idx = load_and_preprocess_ner(time_dir)
    return calc_metric_ner(model_outputs,
                           probs,
                           probs_toks,
                           sampled_probs,
                           sampled_probs_toks,
                           labels,
                           labels_toks,
                           use_idx,
                           methods,
                           metric,
                           level,
                           avg_type)


def load_and_preprocess_ner(time_dir):
    try:
        with open(Path(time_dir) / "dev_inference.json", "r") as f:
            model_outputs = json.loads(f.read())
    except:
        with open(Path(time_dir) / "dev_inference.json", "r") as f:
            model_outputs = stable_json.loads(f.read())

    probs = np.asarray(model_outputs["probabilities"])
    probs_toks = probs.reshape(-1, probs.shape[-1])

    if (
        "sampled_probabilities" in model_outputs.keys()
        and not ("stds" in model_outputs.keys())
        and not ("uncertainty_score" in model_outputs.keys())
    ):
        sampled_probs = np.asarray(model_outputs["sampled_probabilities"])
    elif "sampled_mahalanobis_distance" in model_outputs.keys():
        # mahalanobis case
        # here we transform mahalanobis distance to common shape of NER predictions
        # from (samples, sent_len) to (sampled_probs, samples, sent_len, probas)
        sampled_probs = np.expand_dims(
            np.asarray(model_outputs["sampled_mahalanobis_distance"]), axis=(-1)
        )
        sampled_probs = sampled_probs.reshape(sampled_probs.shape[0], -1, 128, 1)
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
        # TODO: add case for NUQ and work with 3 types of uncertainty in one load
    elif "stds" in model_outputs.keys():
        sampled_probs = np.expand_dims(
            np.asarray(model_outputs["stds"])[:, :, 0], axis=(0, -1)
        )
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
    elif "mahalanobis_distance" in model_outputs.keys() and not (
        "uncertainty_score" in model_outputs.keys()
    ):
        # mahalanobis case
        # here we transform mahalanobis distance to common shape of NER predictions
        # from (samples, sent_len) to (sampled_probs, samples, sent_len, probas)
        sampled_probs = np.expand_dims(
            np.asarray(model_outputs["mahalanobis_distance"]), axis=(0, -1)
        )
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()            
        if "sampled_mahalanobis_distance" in model_outputs.keys():
            sampled_maha = np.expand_dims(
                np.asarray(model_outputs["sampled_mahalanobis_distance"]), axis=(-1)
            )
            sampled_maha = sampled_maha.reshape(
                sampled_maha.shape[0],
                sampled_probs.shape[1],
                sampled_probs.shape[2],
                -1,
            )
            sampled_probs = np.concatenate([sampled_probs, sampled_maha], axis=0)
            model_outputs["sampled_probabilities"] = sampled_probs.tolist()
        elif "mahalanobis_distance_relative" in model_outputs.keys() and not("mahalanobis_distance" in model_outputs.keys()):
            mahalanobis_distance_relative = np.expand_dims(
                np.asarray(model_outputs["mahalanobis_distance_relative"]), axis=(-1)
            )
            mahalanobis_distance = np.expand_dims(
                np.asarray(model_outputs["mahalanobis_distance"]), axis=(-1)
            )
            mahalanobis_distance_marginal = np.expand_dims(
                np.asarray(model_outputs["mahalanobis_distance_marginal"]), axis=(-1)
            )
            sampled_probs = np.stack(
                [mahalanobis_distance, mahalanobis_distance_relative, mahalanobis_distance_marginal],
                axis=0,
            )
            model_outputs["sampled_probabilities"] = sampled_probs.tolist()

    elif "uncertainty_score" in model_outputs.keys():
        sampled_probs = np.expand_dims(
            np.asarray(model_outputs["uncertainty_score"]), axis=(0, -1)
        )
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
    elif "aleatoric" in model_outputs.keys():
        # nuq case - here we have to place all types of uncertainty in sampled_probs
        sampled_probs_aleatoric = np.expand_dims(
            np.asarray(model_outputs["aleatoric"]), axis=(-1)
        )
        sampled_probs_epistemic = np.expand_dims(
            np.asarray(model_outputs["epistemic"]), axis=(-1)
        )
        sampled_probs_total = np.expand_dims(
            np.asarray(model_outputs["total"]), axis=(-1)
        )
        sampled_probs = np.stack(
            [sampled_probs_aleatoric, sampled_probs_epistemic, sampled_probs_total],
            axis=0,
        )
        model_outputs["sampled_probabilities"] = sampled_probs.tolist()
    else:
        raise ValueError(
            f"Error key - dict hasn't either sampled probabilities or mahalanobis distance"
        )
    sampled_probs_toks = sampled_probs.reshape(
        sampled_probs.shape[0], sampled_probs.shape[1] * sampled_probs.shape[2], -1
    )

    labels = np.asarray(model_outputs["true_labels"])
    labels_toks = labels.reshape(-1)

    use_idx = labels_toks != -100
    return (
        model_outputs,
        probs,
        probs_toks,
        sampled_probs,
        sampled_probs_toks,
        labels,
        labels_toks,
        use_idx,
    )


def calc_metric_ner(
    model_outputs,
    probs,
    probs_toks,
    sampled_probs,
    sampled_probs_toks,
    labels,
    labels_toks,
    use_idx,
    methods,
    metric,
    level,
    avg_type='sum'
):
    if metric == "rejection-curve-auc":
        if level == "token":
            return calc_rejection_curve_aucs(
                probs_toks[use_idx],
                labels_toks[use_idx],
                sampled_probs_toks[:, use_idx].transpose(1, 0, 2),
                np.asarray(model_outputs["answers"]).reshape(-1)[use_idx],
                methods=methods,
            )
        return calc_rejection_curve_auc_seq(
            probs,
            labels,
            sampled_probs,
            np.asarray(model_outputs["answers"]),
            methods=methods,
            avg_type=avg_type,
        )

    elif metric == "roc-auc":
        if level == "token":
            return calc_roc_aucs(
                probs_toks[use_idx],
                labels_toks[use_idx],
                sampled_probs_toks[:, use_idx].transpose(1, 0, 2),
                methods=methods,
            )
        return calc_roc_aucs_seq(labels, probs, sampled_probs, methods=methods)
    elif callable(metric):
        return metric(model_outputs, methods=methods, level=level)
    else:
        raise ValueError(f"Error metric {metric}")


def format_results(all_results, baseline_coords=("DPP_last", "max_prob")):
    baseline_row = baseline_coords[0]
    baseline_column = baseline_coords[1]
    baseline = all_results[baseline_row][baseline_column]

    all_formatted_result = {}
    for mc_type, results in all_results.items():
        diff_res = results.drop(columns=baseline_column).subtract(baseline, axis="rows")
        mean_res = diff_res.mean(axis=0)
        std_res = diff_res.std(axis=0)

        diff_final_res = pd.DataFrame.from_records(
            [mean_res, std_res], index=["mean", "std"]
        ).T

        def mean_std_str(row):
            return "{:.1f}±{:.1f}".format(row[0], row[1])

        formatted_results = diff_final_res.apply(mean_std_str, raw=True, axis=1)
        baseline_percent = baseline * 100
        formatted_results.loc["baseline (max_prob)"] = mean_std_str(
            [baseline_percent.mean(), baseline_percent.std()]
        )
        all_formatted_result[mc_type] = formatted_results

    return all_formatted_result


def aggregate_runs(data_path, methods, metric, task_type="classification", oos=False, avg_type='sum'):
    results = []
    model_results = []
    level = None
    for model_seed in os.listdir(data_path):
        try:
            model_seed_int = int(model_seed)
        except:
            if model_seed == "results":
                pass
            else:
                continue

        model_path = Path(data_path) / model_seed

        model_results = []

        for run_seed in os.listdir(model_path):
            run_dir = model_path / run_seed
            try:
                if task_type == "classification":
                    model_results.append(
                        extract_result(run_dir, methods=methods, metric=metric, oos=oos)
                    )
                    inf_time = extract_time(run_dir)
                    model_results[-1].update(inf_time)
                else:
                    level = task_type.split("-")[1]
                    model_results.append(
                        extract_result_ner(
                            run_dir, methods=methods, metric=metric, level=level, avg_type=avg_type
                        )
                    )
            except FileNotFoundError:
                pass
            except:
                continue

        log.info(f"N runs: {len(model_results)}")
        model_avg_res = pd.DataFrame.from_dict(
            model_results, orient="columns"
        )  # .mean(axis=0)
        results.append(model_avg_res)

    results = pd.concat(results, axis=0)
    if level is not None:
        # ner case
        # TODO: changed df structure - now we calc mean by all exps, not by all models. Fix or add switch
        results = results.reset_index(drop=True)
    return results


def calc_rejection_table(
    probabilities,
    labels,
    sampled_probabilities,
    model_answers,
    methods,
    metric,
    ratio_list=None,
):
    if ratio_list is None:
        ratio_list = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    predictions = np.argmax(probabilities, axis=-1)
    errors = (labels != predictions).astype("uint8")

    model_ues = 1 - np.max(probabilities, axis=1)
    sorted_indexes_model = np.argsort(-model_ues)

    results = {}
    model_scores = [
        get_score_ratio(sorted_indexes_model, model_answers, labels, ratio, metric, drop=True)
        for ratio in ratio_list
    ]
    results["max_prob"] = model_scores

    for name, method_function in methods.items():
        ue_scores = method_function(sampled_probabilities)

        ensemble_answers = np.asarray(sampled_probabilities).mean(1).argmax(-1)
        if is_ue_score(name):
            ensemble_answers = predictions
        sorted_indexes_ensemble = np.argsort(-ue_scores)

        ens_scores = [
            get_score_ratio(
                sorted_indexes_ensemble, ensemble_answers, labels, ratio, metric, drop=True
            )
            for ratio in ratio_list
        ]
        results[name] = ens_scores
    results = pd.DataFrame(results).T
    results.columns = [f"{int(ratio*100)}%" for ratio in ratio_list]
    return results


def from_model_outptus_calc_rejection_table(model_outputs, methods,
                                            metric=accuracy_score):
    return calc_rejection_table(
        np.asarray(model_outputs["probabilities"]),
        np.asarray(model_outputs["true_labels"]),
        np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
        np.asarray(model_outputs["answers"]),
        methods=methods,
        metric=metric,
    )


def extract_result_arc_tab(time_dir, methods, metric="roc-auc", oos=False):#from_model_outptus_calc_rejection_table
    with open(Path(time_dir) / "dev_inference.json") as f:
        model_outputs = json.loads(f.read())

    return calc_rejection_table(
        np.asarray(model_outputs["probabilities"]),
        np.asarray(model_outputs["true_labels"]),
        np.asarray(model_outputs["sampled_probabilities"]).transpose(1, 0, 2),
        np.asarray(model_outputs["answers"]),
        methods=methods,
    )


def extract_result_arc_tab_de(model_path, methods, metric="roc-auc"):
    probs = []
    for run_seed in os.listdir(model_path):
        run_dir = model_path / run_seed

        try:
            with open(Path(run_dir) / "dev_inference.json") as f:
                model_outputs = json.loads(f.read())

            probs.append(np.asarray(model_outputs["probabilities"]))

        except FileNotFoundError:
            pass
        except:
            continue

    return calc_rejection_table(
        np.asarray(model_outputs["probabilities"]),
        np.asarray(model_outputs["true_labels"]),
        np.asarray(np.asarray(probs).transpose(1, 0, 2)),
        np.asarray(model_outputs["answers"]),
        methods=methods,
    )


def aggregate_runs_rejection_table(data_path, methods,
                                   metric=from_model_outptus_calc_rejection_table,
                                   de=False):
    results = []
    for model_seed in os.listdir(data_path):
        try:
            model_seed_int = int(model_seed)
        except:
            if model_seed != 'results':
                continue

        model_path = Path(data_path) / model_seed

        if de:
            results.append(extract_result_arc_tab_de(model_path, methods))
            continue

        for run_seed in os.listdir(model_path):
            run_dir = model_path / run_seed
            try:
                #results.append(extract_result_arc_tab(run_dir, methods=methods))
                results.append(extract_result(run_dir, methods=methods, metric=metric))
            except FileNotFoundError:
                pass
            except Exception as e:
                print(e)
                continue

    results = pd.concat(results, axis=0)

    return results


def format_arc_table_results(results, baseline_col, subtract=False, percents=False, ndp=2):
    if subtract:
        baseline = results.T[baseline_col]
        diff_res = results.T.drop(columns=baseline_col).subtract(baseline, axis="rows")
        diff_res = pd.concat([baseline], axis=1).T
    else:
        diff_res = results

    mean_res = results.groupby(level=0).mean()
    std_res = results.groupby(level=0).std()
    if percents:
        mean_res *= 100
        std_res *= 100
    formatted_results = mean_res.applymap(
        lambda x: "{}±".format(round(x, ndp))
    ) + std_res.applymap(lambda x: "{}".format(round(x, ndp)))

    return formatted_results

def mean_std_str(row, ndp):
    if ndp == 2:
        return "{:.2f}±{:.2f}".format(round(row[0], ndp), round(row[1], ndp))
    return "{}±{}".format(round(row[0], ndp), round(row[1], ndp))


def format_results2(results, percents=False, ndp=2):
    """ndp: number of decimal points"""
    mean_res = results.mean(axis=0)
    std_res = results.std(axis=0)
    final_results = pd.DataFrame.from_records(
        [mean_res, std_res], index=["mean", "std"]
    ).T

    if percents:
        final_results *= 100.0

    formatted_results = final_results.apply(
        lambda row: mean_std_str(row, ndp), raw=True, axis=1
    )
    return formatted_results


def improvement_over_baseline(
    results,
    baseline_col,
    baseline=None,
    metric="roc-auc",
    subtract=False,
    percents=False,
):
    if baseline is None:
        baseline = results[baseline_col]
        if subtract:
            diff_res = results.drop(columns=baseline_col).subtract(
                baseline, axis="rows"
            )
        else:
            diff_res = results.drop(columns=baseline_col)
    else:
        baseline_raw = baseline[metric]
        baseline = results[baseline_col]
        if subtract:
            diff_res = results.drop(columns=baseline_col) - baseline_raw.values[0]
        else:
            diff_res = results.drop(columns=baseline_col)

    ndp = 2# if metric == "rejection-curve-auc" else 1
    formatted_results = format_results2(diff_res, percents=percents, ndp=ndp)

    if percents:
        baseline_percent = baseline * 100
    else:
        baseline_percent = baseline
    formatted_results.loc["baseline (max_prob)"] = mean_std_str(
        [baseline_percent.mean(), baseline_percent.std()], ndp
    )
    formatted_results.loc["count"] = baseline_percent.shape[0]
    return formatted_results


def get_model_type(model_path):
    # model_type = model_path.split('/')[-2]
    if "bert" in model_path:
        return "bert"
    elif "electra" in model_path:
        return "electra"
    else:
        return "unknown"


def collect_configs(dir_name):
    cfg_str = ""
    for model_seed in os.listdir(dir_name):
        model_path = Path(dir_name) / model_seed
        for run_seed in os.listdir(model_path):
            run_path = model_path / run_seed

            with open(run_path / ".hydra" / "config.yaml") as f:
                cfg = yaml.load(f, Loader=Loader)

            if cfg is None:
                continue

            # print(type(cfg))
            cfg_str_new = "_".join(
                str(e)
                for e in (
                    cfg["ue"]["dropout_type"],
                    cfg["ue"]["dropout"]["dry_run_dataset"],
                    cfg["ue"]["dropout"]["mask_name"],
                    cfg["ue"]["dropout"]["max_frac"],
                    get_model_type(cfg["model"]["model_name_or_path"]),
                    cfg["data"]["task_name"],
                )
            )

            if cfg_str:
                if cfg_str != cfg_str_new:
                    print("Error, different cfg_strs:", cfg_str, cfg_str_new)

            cfg_str = cfg_str_new

    return cfg_str