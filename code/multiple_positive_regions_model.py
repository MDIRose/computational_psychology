import baseline_model as baseline
from utility_functions import split_number_regions

from typing import List


def generalize(consequential_region: int, positive_samples: List[int],  n_region: int) -> List[float]:
    """
    :param consequential_region: size of consequential region
    :param positive_samples: list of positive examples in consequential region
    :param n_regions: split into n_regions
    :return: list of posterior probabilities for each discrete value in consequential region
    """
    regions = split_number_regions(
        consequential_region, positive_samples, n_region)
    positive_prob = baseline.generalize(consequential_region, [], [])
    for region in regions:
        prob = baseline.generalize(consequential_region, region, [])
        positive_prob = [max(positive_prob[i], prob[i])
                         for i in range(len(positive_prob))]
    return positive_prob
