# This code is based on homework 2 (which itself is based on code written originally by Danielle Navarro)

import baseline_model as baseline

from typing import List


def generalize(consequential_region: int, positive_samples: List[int], negative_samples: List[int]) -> List[float]:
    """
    :param consequential_region: size of consequential region
    :param positive_samples: list of positive examples in consequential region
    :param negative_samples: list of negative examples in consequential region (unused)
    :return: list of posterior probabilities for each discrete value in consequential region
    """
    positive_prob = baseline.generalize(consequential_region, positive_samples, [])
    negative_prob = baseline.generalize(consequential_region, negative_samples, [])
    generalization = [positive_prob[i] - negative_prob[i] for i in range(len(positive_prob))]
    return [0 if x <= 0 else x for x in generalization]