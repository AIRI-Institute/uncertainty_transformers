from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, auc, roc_auc_score

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import re

from ue4nlp.ue_scores import *


default_methods = {
    "bald": bald,
    "var_ratio": var_ratio,
    "entropy": mean_entropy,
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


def get_score_ratio(sorted_indexes, answers, true_answers, ratio):
    last_index = int(len(sorted_indexes) * ratio)
    sel_indexes = sorted_indexes[:last_index]
    unsel_indexes = sorted_indexes[last_index:]

    sel_answers = true_answers[sel_indexes].tolist() + answers[unsel_indexes].tolist()
    sel_true_answers = (
        true_answers[sel_indexes].tolist() + true_answers[unsel_indexes].tolist()
    )
    score = accuracy_score(sel_true_answers, sel_answers)
    return score


def plot_error_detection(
    probabilities,
    labels,
    sampled_probabilities,
    stds=None,
    methods=None,
    use_means=False,
    verbose=True,
):
    """
    N - number of points in the dataset, C - number of classes, R - number of sampling runs
    all arguments expect to be np.array
    :param probabilities:  probabilities by model without dropout, NxC
    :param labels: true labels for classification, N
    :param sampled_probabilities: probabilities sampled by dropout, NxRxC
    :return: None, make roc curve plot for error detection
    """
    if methods is None:
        methods = default_methods

    predictions = np.argmax(probabilities, axis=-1)
    errors = (labels != predictions).astype("uint8")

    if verbose:
        plt.figure(dpi=150)
    auc = []

    for name, method_function in methods.items():
        if not use_means:
            ue_scores = method_function(sampled_probabilities)
        else:
            ue_scores = method_function(sampled_probabilities, probabilities)

        fpr, tpr, _ = roc_curve(errors, ue_scores)

        if verbose:
            plt.plot(fpr, tpr, label=name)
            print(f"{name}:", roc_auc_score(errors, ue_scores))
        auc.append(roc_auc_score(errors, ue_scores))

    if stds is not None:
        ue_scores = stds.mean(1)  # [:, 1]
        fpr, tpr, _ = roc_curve(errors, ue_scores)
        if verbose:
            plt.plot(fpr, tpr, label="SNGP")
            print(f"SNGP:", roc_auc_score(errors, ue_scores))
        auc.append(roc_auc_score(errors, ue_scores))

    max_prob = 1 - np.max(probabilities, axis=-1)
    fpr, tpr, _ = roc_curve(errors, max_prob)
    if verbose:
        print(f"max_prob:", roc_auc_score(errors, max_prob))
    auc.append(roc_auc_score(errors, max_prob))

    if verbose:
        plt.plot(fpr, tpr, label="max_prob")
        plt.ylabel("True positive rate", fontdict={"size": 5})
        plt.xlabel("False positive rate", fontdict={"size": 5})
        plt.legend()

    return np.asarray(auc)


def plot_rejection_curve_aucs(
    probabilities,
    labels,
    sampled_probabilities,
    model_answers,
    stds=None,
    methods=None,
    use_means=False,
    verbose=True,
):
    """
    N - number of points in the dataset, C - number of classes, R - number of sampling runs
    all arguments expect to be np.array
    :param probabilities:  probabilities by model without dropout, NxC
    :param labels: true labels for classification, N
    :param sampled_probabilities: probabilities sampled by dropout, NxRxC
    :param stds:  stds by model with SNGP, NxC
    :return: None, make roc curve plot for error detection
    """
    if methods is None:
        methods = default_methods

    predictions = np.argmax(probabilities, axis=-1)
    errors = (labels != predictions).astype("uint8")

    if verbose:
        plt.figure(dpi=150)
    aucs = []

    ratio_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for name, method_function in methods.items():
        if not use_means:
            ue_scores = method_function(sampled_probabilities)
        else:
            ue_scores = method_function(sampled_probabilities, probabilities)

        ensemble_answers = np.asarray(sampled_probabilities).mean(1).argmax(-1)
        sorted_indexes_ensemble = np.argsort(-ue_scores)
        ens_scores = [
            get_score_ratio(sorted_indexes_ensemble, ensemble_answers, labels, ratio)
            for ratio in ratio_list
        ]

        if verbose:
            plt.plot(ratio_list, ens_scores, label=name, linewidth=1)
            print(f"{name}:", auc(ratio_list, ens_scores))
        aucs.append(auc(ratio_list, ens_scores))

    if stds is not None:
        ensemble_answers = np.asarray(model_answers)
        ue_scores = stds[:, 1]
        sorted_indexes = np.argsort(-ue_scores)
        std_scores = [
            get_score_ratio(sorted_indexes, model_answers, labels, ratio)
            for ratio in ratio_list
        ]
        if verbose:
            plt.plot(ratio_list, std_scores, label="SNGP", linewidth=1)
            print(f"SNGP:", auc(ratio_list, std_scores))
        aucs.append(auc(ratio_list, std_scores))

    model_ues = 1 - np.max(probabilities, axis=1)
    sorted_indexes_model = np.argsort(-model_ues)
    model_scores = [
        get_score_ratio(sorted_indexes_model, model_answers, labels, ratio)
        for ratio in ratio_list
    ]
    if verbose:
        print(f"max_prob:", auc(ratio_list, model_scores))
    aucs.append(auc(ratio_list, model_scores))

    if verbose:
        plt.plot(ratio_list, model_scores, label="max_prob", linewidth=1)
        plt.ylabel("True positive rate", fontdict={"size": 5})
        plt.xlabel("False positive rate", fontdict={"size": 5})
        plt.legend()
    return np.asarray(aucs)


def get_score_ratio(sorted_indexes, answers, true_answers, ratio):
    last_index = int(len(sorted_indexes) * ratio)
    sel_indexes = sorted_indexes[:last_index]
    unsel_indexes = sorted_indexes[last_index:]

    sel_answers = true_answers[sel_indexes].tolist() + answers[unsel_indexes].tolist()
    sel_true_answers = (
        true_answers[sel_indexes].tolist() + true_answers[unsel_indexes].tolist()
    )
    score = accuracy_score(sel_true_answers, sel_answers)
    return score



def plot_error_detection_seq(labels, probs, sampled_probs, methods=None):

    sampled_probs, probs, predictions, labels = unpad_preds(
        probs, sampled_probs, np.argmax(probs, axis=-1), labels
    )

    if methods is None:
        methods = default_methods

    errors = [1.0 * (l != p) for l, p in zip(labels, predictions)]

    plt.figure(dpi=150)
    for name, method_function in methods.items():
        ue_scores = seq_ue(sampled_probs, method_function)
        fpr, tpr, _ = roc_curve(errors, ue_scores)
        print(f"{name}: {roc_auc_score(errors, ue_scores)}")
        plt.plot(fpr, tpr, label=name)

    n_examples = len(errors)
    ue_scores_max = np.zeros(n_examples)
    for i in range(n_examples):
        sent = probs[i]
        true_probs_max = np.asarray([np.max(proba) for proba in sent])
        ue_scores_max[i] = np.mean(1 - true_probs_max)

    fpr, tpr, _ = roc_curve(errors, ue_scores_max)
    print(f"max_prob:", roc_auc_score(errors, ue_scores_max))

    plt.plot(fpr, tpr, label="max_prob")
    plt.ylabel("True positive rate", fontdict={"size": 5})
    plt.xlabel("False positive rate", fontdict={"size": 5})
    plt.legend()


def draw_charts(charts, save_dir=None, save_name=None):
    color_pool = [
        ("red", "darksalmon", "o"),
        ("midnightblue", "skyblue", "v"),
        ("g", "lightgreen", "s"),
        # ('gold', 'palegoldenrod', '*'),
        ("maroon", "rosybrown", "*"),
        ("purple", "violet", "+"),
        ("slategrey", "lightgrey", "1"),
        ("darkorange", "wheat", "s"),
        ("darkcyan", "lightcyan", "P"),
    ]

    plt.figure(dpi=150)

    plt_list = []
    for i, chart in enumerate(charts):
        chart, label = chart

        x = [e[0] for e in chart]
        y = [e[1] for e in chart]

        color_line = color_pool[i][0]
        color_bg = color_pool[i][1]
        marker = color_pool[i][2]

        # chart = [(f, np.asarray(s).mean()) for f, s in roc_aucs.items()]
        means = np.asarray([np.asarray(e).mean() for e in y])
        disp = np.asarray([np.asarray(e).std() for e in y])

        line_thikness = 0.4
        plt.fill_between(x, means - disp, means + disp, color=color_bg, alpha=0.3)
        # plt.xticks(range(1, means.shape[0] + 2, 2))

        plt_list.append(
            plt.plot(
                x,
                means,
                color=color_line,
                label=label,
                marker=marker,
                linewidth=line_thikness,
                markersize=2,
            )[0]
        )

    plt.ylabel("roc_auc")
    plt.xlabel("data_frac")
    plt.legend(handles=plt_list, loc="lower left", fontsize="x-large")

    if save_dir is not None and save_name is not None:
        plt.savefig(Path(save_dir) / f"{save_name}.pdf")

    plt.show()


def roc_auc_error_detection(ue_scores, answers, y):
    sorted_indexes = np.argsort(np.array(ue_scores))

    #     print(answers)
    #     print(len(answers), len(y), len(ue_scores))

    uncertainties_sorted = [ue_scores[e] for e in sorted_indexes]
    y_sorted = [y[e] for e in sorted_indexes]
    answers_sorted = [answers[e] for e in sorted_indexes]

    #     print(len(answers_sorted), len(y_sorted))
    binary_y = [y_o != a_o for y_o, a_o in zip(y_sorted, answers_sorted)]

    return roc_auc_score(binary_y, np.array(uncertainties_sorted))


def create_chart_data(session_dir, f_ue_score):
    roc_aucs = {}
    for filename in os.listdir(session_dir):
        if not re.match("\d+\.\d+", filename):
            continue

        frac = float(filename)

        frac_dir = Path(session_dir) / filename

        frac_scores = []
        for date_filename in os.listdir(frac_dir):
            date_dir = Path(frac_dir) / date_filename

            for time_filename in os.listdir(date_dir):
                time_dir = Path(date_dir) / time_filename

                try:
                    with open(Path(time_dir) / "dev_inference.json") as f:
                        model_outputs = json.load(f)

                    ue_scores = f_ue_score(model_outputs)

                    roc_auc = roc_auc_error_detection(
                        ue_scores=ue_scores,
                        answers=model_outputs["answers"],
                        y=model_outputs["true_labels"],
                    )
                    frac_scores.append(roc_auc)
                except FileNotFoundError:
                    pass

        roc_aucs[frac] = frac_scores

    chart = sorted([(k, v) for k, v in roc_aucs.items()], key=lambda e: e[0])
    chart = [e for e in chart if e[1]]

    return chart
