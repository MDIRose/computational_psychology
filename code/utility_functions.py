import numpy as np
import math

from typing import List

# Take input [1, 100] and  negative input [50] generates bimodal disjoint model

def split_example_regions(positive_examples: List[int],negative_examples: List[int]) -> List[List[int]]:
    """
    This function generates disjoint cluster hypothesis based on observed samples
    :param positive_examples:
    :param negative_examples:
    :return:
    """
    positive_examples = sorted(positive_examples)
    negative_examples = sorted(negative_examples)
    regions = []
    region = []
    i = 0
    for example in positive_examples:
        if len(negative_examples) <= i or example <  negative_examples[i]:
            region.append(example)
        else:
            regions.append(region)
            region = [example]
            i += 1
    regions.append(region)
    return regions
# [1, 50, 100] 3 -> [[1], [50], [100]]
def split_number_regions(consequential_region: int, positive_examples: List[int], n_regions: int) -> List[List[int]]:
    if len(positive_examples) < n_regions:
        raise 'Too many regions for number of examples'
    split_distance = consequential_region / n_regions
    regions = [[] for _ in range(n_regions)]
    i = 1
    for example in positive_examples:
        if example >= split_distance*i:
            i +=1
        regions[i-1].append(example)
    return regions

split_number_regions(120, [1, 50, 100], 2)

