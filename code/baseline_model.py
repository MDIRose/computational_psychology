# This code is based on homework 2 (which itself is based on code written originally by Danielle Navarro)

import math
import numpy as np

from typing import List


def generalize(consequential_region: int, samples: List[int]) -> List[float]:
    """
    :param consequential_region: size of consequential region
    :param samples: list of positive examples in consequential region
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

    # Create a matrix containing likelihoods for all
    # possible values between 1 and 100, as specified
    # by all 5050 hypotheses
    likelihood = np.zeros((nH, consequential_region))
    for h in range(0, nH):
        # TODO: Write code in this for loop that fills in
        # the likelihood for each hypothesis. Row i contains
        # the likelihood values for all integers for hypothesis i.
        #
        # A couple things to keep in mind:
        # 1. Remember that python lists start indexing at 0 even
        #    though we want to record likelihoods for values
        #    starting at 1.
        # 2. Remember that we are recording logarithms of values
        #    so you should be storing logs of likelihoods.
        rH = hypotheses[h, :]
        lbH = rH[0]
        ubH = rH[1]
        for i in range(0, consequential_region):
            # We assume strong sampling
            v = i+1
            if lbH <= v <= ubH:
                likelihood[h, i] = math.log(float(1) / ((ubH-lbH)+1))
            else:
                likelihood[h, i] = 0

    # Likelihood of the observed data
    posterior = prior  # Initialize the posterior at the prior
    for obs in samples:
        # Update the posterior
        for h in range(0, nH):
            # The log of 0 is undefined, but we are later going to
            # convert back to probabilities by applying the exponential
            # function exp(), so for cases where the probability should
            # be 0, we will artificially set the posterior to -1000
            # because exp(-1000) ~= 0.

            # Note that I subtract 1 from observation. See Note 1 above.
            # You are free to change this if you write the code differently.
            if likelihood[h, obs-1] == 0:
                posterior[h] = -1000

            # Otherwise, we want to multiply the prior and the likelihood.
            # Because we are dealing with logarithms, however, we add the
            # logs together instead.
            else:
                posterior[h] = prior[h] + likelihood[h, obs-1]

    # Convert posteriors to probabilities
    posterior_sum = 0
    for h in range(0, nH):
        posterior[h] = math.exp(posterior[h])

        # Keep a running sum of the posteriors
        posterior_sum = posterior_sum + posterior[h]

    # Normalize the probabilities so that they sum to 1
    # Because we didn't compute the marginal likelihood p(x), we have
    # generated a true probability distribution: our probabilities do not sum to
    # 1. But we can fix this by simply "normalizing" the distribution. We
    # accomplish this by dividing every posterior probability by the sum of
    # probabilities. This is acceptable because it preserves all the relative
    # probabilities -- in other words, we are just scaling all the values by
    # a constant.
    for h in range(0, nH):
        posterior[h] = posterior[h] / posterior_sum

        # What is the probability that a particular value is
    # in the consequential region?
    answer = np.zeros(consequential_region)
    for i in range(0, consequential_region):
        a = 0
        # Select all hypotheses with boundary 1 <=
        # value i+1 and boundary 2 >= i+1. That is,
        # all hypotheses with boundaries that include i+1.
        for h in range(0, nH):
            rH = hypotheses[h, :]
            lbH = rH[0]
            ubH = rH[1]
            v = i + 1
            if lbH <= v and v <= ubH:
                a += posterior[h]

        answer[i] = a

    return answer.tolist()