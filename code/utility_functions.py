import numpy as np
import math

from typing import List

# Take input [1, 100] and  negative input [50] generates bimodal disjoint model

def split_example_regions(positive_examples: List[int],negative_examples: List[int]) -> List[List[int]]:
    """
    This function generates disjoint cluster hypothesis based on observed samples
    :param consequential_region:
    :param positive_examples:
    :param negative_examples:
    :return:
    """
    positive_examples = sorted(positive_examples)
    negative_examples = sorted(negative_examples)
    regions = []

    if len(negative_examples) == 0:
        return [[positive_examples[0], positive_examples[-1]]]

    for example in negative_examples:
        values = list(filter(lambda x: x < example, positive_examples))
        lower = values[0]
        if len(regions) == 0:
            lower = 0
        regions.append([lower, example-1])

    final_values = list(filter(lambda x:  x > negative_examples[-1], positive_examples))
    if len(final_values)  != 0:
        lower = min(regions[-1][-1]+1, final_values[0])
        regions.append([lower, 100])
    else:
        regions[-1][-1] = 100
    return regions

print(split_example_regions([1,  100], [50]))