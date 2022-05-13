import math
import numpy as np

from typing import List


def generalize(consequential_region: int, positive_samples: List[int], negative_samples: List[int]) -> List[float]:
    """
    This variant of the model removes hypotheses from the hypothesis space which are known as negative examples.
    This code is based on homework 2 (which itself is based on code written originally by Danielle Navarro).
    :param consequential_region: size of consequential region
    :param positive_samples: list of positive examples in consequential region
    :param negative_samples: list of negative examples in consequential region
    :return: list of posterior probabilities for each discrete value in consequential region
    """
    nH = math.comb(consequential_region+1, 2)
    # Create the hypothesis space: Every hypothesis is continuous and defined by an upper and lower bound
    hypotheses = np.zeros((nH, 2))
    lower_bound = 1
    upper_bound = lower_bound
    for h in range(0, nH):
        hypotheses[h, :] = [lower_bound, upper_bound]
        upper_bound += 1
        if upper_bound > consequential_region:
            lower_bound += 1
            upper_bound = lower_bound

    # Specify prior: All hypothesis are equally likely
    prior = np.repeat(math.log(1.0/nH), nH)

    # Generate likelihood values (stored in log format)
    likelihood = np.zeros((nH, consequential_region))
    for h in range(0, nH):
        rH = hypotheses[h, :]
        lbH = rH[0]
        ubH = rH[1]
        excluded = any(map(lambda n: lbH <= n <= ubH, negative_samples))
        for i in range(0, consequential_region):
            # We assume strong sampling
            v = i+1
            if (lbH <= v <= ubH) and not excluded:
                likelihood[h, i] = math.log(float(1) / ((ubH-lbH)+1))
            else:
                likelihood[h, i] = 0

    # Likelihood of the observed data
    posterior = prior  # Initialize the posterior at the prior
    for obs in positive_samples:
        # Update the posterior
        for h in range(0, nH):
            if likelihood[h, obs-1] == 0:
                # exp(-1000) ~= 0.
                posterior[h] = -1000
            else:
                # multiply the prior and the likelihood in log representation
                posterior[h] = prior[h] + likelihood[h, obs-1]

    # Convert log posteriors to actual probabilities
    posterior_sum = 0
    for h in range(0, nH):
        posterior[h] = math.exp(posterior[h])

        # Keep a running sum of the posteriors
        posterior_sum = posterior_sum + posterior[h]

    # Normalize the probabilities so that they sum to 1
    for h in range(0, nH):
        posterior[h] = posterior[h] / posterior_sum

    # assign probabilities to values in consequential region
    answer = np.zeros(consequential_region)
    for i in range(0, consequential_region):
        a = 0
        for h in range(0, nH):
            rH = hypotheses[h, :]
            lbH = rH[0]
            ubH = rH[1]
            v = i + 1
            if lbH <= v <= ubH:
                a += posterior[h]
        answer[i] = a

    return answer.tolist()