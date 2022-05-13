import baseline_model as baseline
import numpy as  np
from utility_functions import split_number_regions

from typing import List


def generalize(consequential_region: int, positive_samples: List[int],  n_region: int) -> List[float]:
    """
    :param consequential_region: size of consequential region
    :param positive_samples: list of positive examples in consequential region
    :param n_regions: split into n_regions
    :return: list of posterior probabilities for each discrete value in consequential region
    """
    regions = split_number_regions(positive_samples, n_region)
    positive_prob = [0 for _ in range(consequential_region)]
    for region in regions:
        prob = baseline.generalize(consequential_region, region, [])
        positive_prob = [max(positive_prob[i], prob[i])
                         for i in range(len(positive_prob))]
    return positive_prob

def generalize_softmax(consequential_region: int, positive_samples: List[int],  n_region: int) -> List[float]:
    """
    :param consequential_region: size of consequential region
    :param positive_samples: list of positive examples in consequential region
    :param n_regions: split into n_regions
    :return: list of posterior probabilities for each discrete value in consequential region
    """
    regions = split_number_regions(positive_samples, n_region)
    positive_prob = [0 for _ in range(consequential_region)]
    for region in regions:
        prob = baseline.generalize(consequential_region, region, [])
        positive_prob = [positive_prob[i]+prob[i]
                         for i in range(len(positive_prob))]
    final_prob = []
    final_sum = 0
    for prob in positive_prob:
        final_sum = final_sum + np.exp(prob)
    for prob in positive_prob:
        final_prob.append(np.exp(prob)/final_sum)
    return final_prob

consequential_region = 120
positive_examples = [1,100]
negative_examples = [50]
generalize(consequential_region,positive_examples, 2)
