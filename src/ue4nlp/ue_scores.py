import numpy as np


def entropy(x):
    return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)


def mean_entropy(sampled_probabilities):
    return entropy(np.mean(sampled_probabilities, axis=1))


def bald(sampled_probabilities):
    predictive_entropy = entropy(np.mean(sampled_probabilities, axis=1))
    expected_entropy = np.mean(entropy(sampled_probabilities), axis=1)

    return predictive_entropy - expected_entropy


def var_ratio(sampled_probabilities):
    top_classes = np.argmax(sampled_probabilities, axis=-1)
    # count how many time repeats the strongest class
    mode_count = lambda preds: np.max(np.bincount(preds))
    modes = [mode_count(point) for point in top_classes]
    ue = 1.0 - np.array(modes) / sampled_probabilities.shape[1]
    return ue


def sampled_max_prob(sampled_probabilities):
    mean_probabilities = np.mean(sampled_probabilities, axis=1)
    top_probabilities = np.max(mean_probabilities, axis=-1)
    return 1 - top_probabilities


def probability_variance(sampled_probabilities, mean_probabilities=None):
    if mean_probabilities is None:
        mean_probabilities = np.mean(sampled_probabilities, axis=1)

    mean_probabilities = np.expand_dims(mean_probabilities, axis=1)

    return ((sampled_probabilities - mean_probabilities) ** 2).mean(1).sum(-1)
    # return ((sampled_probabilities - mean_probabilities)**2).mean(1).max(-1)


def seq_ue(sampled_probabilities, method_function, avg_type="sum", compute_func=True):
    n_examples = len(sampled_probabilities)

    if avg_type == "sum":
        avg_method = np.sum
    elif avg_type == "mean":
        avg_method = np.mean
    elif avg_type == "max":
        avg_method = np.max

    ue_scores = np.zeros(n_examples)

    for i in range(n_examples):
        sent = np.asarray(sampled_probabilities[i])
        if compute_func:
            ue_scores[i] = avg_method(method_function(sent.transpose(0, 2, 1)))
        else:
            #mc mahalanobis case
            ue_scores[i] = avg_method(sent.transpose(0, 2, 1).max(0), axis=0)

    return ue_scores
