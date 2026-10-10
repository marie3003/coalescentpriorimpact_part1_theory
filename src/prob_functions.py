"""
Likelihood and coalescent priors for the tMRCA of two sequences sampled at the same time.

Time is measured backwards from the present and the generation time is assumed to be 1, so
population sizes and times share the same unit (e.g. years).
"""

from math import exp, lgamma
import numpy as np


def likelihood_tMRCA_mutations(k_mut, mu, t_MRCA, L=100):
    """
    Calculate the likelihood of observing k_mut mutations given a tMRCA and mutation rate.
    We only look at the tMRCA of two sequences sampled at the same time point, so mutations
    accumulate along both branches: k_mut ~ Poisson(2 * mu * L * t_MRCA).

    Parameters:
    k_mut (int): Number of mutations observed.
    mu (float): Mutation rate per base pair per generation.
    t_MRCA (float): calendar time to the most recent common ancestor, assume a generation time of 1.
    L (int): Length of the genome in base pairs (default is 100).

    Returns:
    float: Likelihood of observing k_mut mutations.
    """

    expected_mutations = 2 * mu * L * t_MRCA

    if expected_mutations <= 0:
        return 0.0

    # Compute in log space to avoid overflow of λ^k and k! for large k:
    # log(P) = k*log(λ) - λ - log(k!)
    log_likelihood = (
        k_mut * np.log(expected_mutations)
        - expected_mutations
        - lgamma(k_mut + 1)  # log(k!)
    )
    return np.exp(log_likelihood)


def coalescent_prior(t_MRCA, N):
    """
    Calculate the prior probability of a tMRCA given a constant population size N.
    For two lineages the coalescence time is exponentially distributed with rate 1/N.

    Parameters:
    t_MRCA (float): calendar time to the most recent common ancestor.
    N (int): Effective population size.

    Returns:
    float: Prior probability of the tMRCA.
    """

    if t_MRCA < 0:
        return 0.0

    prior = 1 / N * exp(-t_MRCA / N)

    return prior


def coalescent_prior_expN_present(t_MRCA, N_present, beta):
    """
    Calculate the prior probability of a tMRCA given an exponentially growing population,
    N(t) = N_present * exp(-beta * t) with t = 0 at the present.

    The coalescence rate is 1/N(t) = exp(beta * t) / N_present, so
    p(t) = exp(beta * t) / N_present * exp(-(exp(beta * t) - 1) / (N_present * beta)).

    Parameters:
    t_MRCA (float): calendar time to the most recent common ancestor.
    N_present (int): Effective population size at the present time.
    beta (float): Growth rate of the population (positive = growth, negative = decline).

    Returns:
    float: Prior probability of the tMRCA.
    """

    if t_MRCA < 0:
        return 0.0

    # Coalescence rate at time t_MRCA (up to the 1/N_present factor)
    exp1 = exp(beta * t_MRCA)

    # Probability of no coalescence before t_MRCA
    inner_term = (exp(beta * t_MRCA) - 1)
    exp2 = exp(-(1 / (N_present * beta)) * inner_term)

    prior = (1 / N_present) * exp1 * exp2

    return prior


def coalescent_prior_bottleneck(t_MRCA, N_high, N_low, t_bottleneck_start, t_bottleneck_end):
    """
    Calculate the prior probability of a tMRCA given a piecewise constant population size:
    N_high outside the bottleneck and N_low during [t_bottleneck_start, t_bottleneck_end].

    Parameters:
    t_MRCA (float): calendar time to the most recent common ancestor.
    N_high (int): Population size outside the bottleneck.
    N_low (int): Population size during the bottleneck.
    t_bottleneck_start (float): Start time of the bottleneck backwards in time.
    t_bottleneck_end (float): End time of the bottleneck backwards in time.

    Returns:
    float: Prior probability of the tMRCA.
    """

    if t_MRCA < 0:
        return 0.0

    # In each interval: (coalescence rate at t_MRCA) * exp(-integrated rate from 0 to t_MRCA)
    if t_MRCA < t_bottleneck_start:
        # More recent than the bottleneck
        prior = (1 / N_high) * exp(-t_MRCA / N_high)
    elif t_bottleneck_start <= t_MRCA <= t_bottleneck_end:
        # During the bottleneck
        prior = (1 / N_low) * exp(-(t_bottleneck_start / N_high + (t_MRCA - t_bottleneck_start) / N_low))
    else:
        # Further back than the bottleneck
        prior = (1 / N_high) * exp(-(t_bottleneck_start / N_high + (t_bottleneck_end - t_bottleneck_start) / N_low + (t_MRCA - t_bottleneck_end) / N_high))

    return prior
