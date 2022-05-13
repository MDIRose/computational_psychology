import math
import numpy as np

import elimination_model as elimination

from typing import List


def generalize(consequential_region: int, positive_samples: List[int], negative_samples: List[int]) -> List[float]:
    """
    This variant of the model handles discontinuous hypotheses by using negative samples to split the hypothesis space.
    :param consequential_region: size of consequential region
    :param positive_samples: nonempty list of positive examples in consequential region
    :param negative_samples: list of negative examples in consequential region
    :return: list of posterior probabilities for each discrete value in consequential region
    """
    # input validation
    if len(positive_samples) < 1:
        raise 'List of positive samples must be nonempty'
    positive_samples = sorted(positive_samples)

    # split positive samples into sections using negative samples as separators
    low = positive_samples[0]
    section = [low]
    positive_sections = [section]
    for i in range(1, len(positive_samples)):
        high = positive_samples[i]
        if any_in_range(negative_samples, low, high):
            section = [high]
            positive_sections.append(section)
        else:
            section.append(high)
        low = high

    # calculate per-section probabilities
    num_sections = len(positive_sections)
    section_probabilities = np.zeros((num_sections, consequential_region))
    for i, section in enumerate(positive_sections):
        section_probabilities[i] = elimination.generalize(consequential_region, section, negative_samples)

    # sum section probabilities for output
    return section_probabilities.sum(axis=0).tolist()

def any_in_range(items: List[int], low: int, high: int) -> bool:
    return any(map(lambda n: low <= n <= high, items))

consequential_region = 120
positive_examples = [0,100]
negative_examples = [50]
generalize(consequential_region,positive_examples,negative_examples )
