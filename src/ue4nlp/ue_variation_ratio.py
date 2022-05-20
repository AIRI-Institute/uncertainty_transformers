import numpy as np

from collections import Counter
from typing import Iterable, Union


def find_most_common(row: Iterable[str], mode: Union["elem", "count"]):
    """
    Given iterable of words, return either most common element or its count
    """
    if mode == "elem":
        return Counter(row).most_common(1)[0][0]
    elif mode == "count":
        return Counter(row).most_common(1)[0][1]


def ue_variation_ratio(answers):
    answers = [np.array(e, dtype=object) for e in answers]
    answers = np.stack(answers, -1)

    scores = 1.0 - np.array(
        [find_most_common(ans, "count") / answers.shape[1] for ans in answers]
    )
    return scores
