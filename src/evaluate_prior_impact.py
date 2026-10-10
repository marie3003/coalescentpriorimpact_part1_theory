"""
Build DataFrames of prior influence metrics over grids of population models and mutational signals.

For every parameter combination the tMRCA likelihood, coalescent prior and (unnormalized)
posterior are evaluated on a grid t_vec = linspace(0, t_max, 10 * t_max). The posterior under the
coalescent prior is then compared to the posterior under a flat prior (= the likelihood).
"""

import numpy as np
import pandas as pd
from src.prob_functions import likelihood_tMRCA_mutations, coalescent_prior, coalescent_prior_expN_present, coalescent_prior_bottleneck
from src.metrics import calculate_rel_median_shift, calculate_median_shift, calculate_wasserstein_dist


def _time_grid(t_max):
    """Time grid with 10 points per time unit on which all densities are evaluated."""
    return np.linspace(0, t_max, int(t_max * 10))


def _likelihoods(k, mu, L, t_vec):
    """Likelihood of observing k mutations for every tMRCA in t_vec."""
    return np.array([likelihood_tMRCA_mutations(k, mu, t, L) for t in t_vec])


def _prior_influence_metrics(t_vec, posteriors, likelihoods, abs_value):
    """
    Compare the posterior under the coalescent prior to the posterior under a flat prior.
    Signed shifts (abs_value=False) are positive if the coalescent prior pushes the tMRCA further
    into the past.
    """
    return {
        'rel_median_shift': calculate_rel_median_shift(t_vec, posteriors, likelihoods, abs_value=abs_value),
        'median_shift': calculate_median_shift(t_vec, posteriors, likelihoods, abs_value=abs_value),
        'wasserstein_dist': calculate_wasserstein_dist(t_vec, posteriors, likelihoods),
    }


################ Pathogen / mutational signal comparison ################

def create_prior_influence_metric_df_const_pathogen(pathogen_params):
    """
    Create a DataFrame of prior influence metrics for different pathogens (mutational signals)
    and different constant population sizes N. Shifts are reported as absolute values.

    Parameters:
    pathogen_params (dict): {pathogen name: {'N_values', 'mu', 'k', 'L', 't_max'}} with the
        population sizes to evaluate, mutation rate, number of mutations, genome length, and
        maximum tMRCA.

    Returns:
    pd.DataFrame: One row per (pathogen, N).
    """

    data = []

    for name, params in pathogen_params.items():
        t_vec = _time_grid(params['t_max'])
        likelihoods = _likelihoods(params['k'], params['mu'], params['L'], t_vec)

        for N in params['N_values']:
            priors = np.array([coalescent_prior(t, N) for t in t_vec])
            posteriors = likelihoods * priors

            data.append({
                'pathogen': name,
                'N': N,
                **_prior_influence_metrics(t_vec, posteriors, likelihoods, abs_value=True),
            })

    return pd.DataFrame(data)


def create_prior_influence_metric_df_exp_pathogen(pathogen_params, N, beta_vec):
    """
    Create a DataFrame of prior influence metrics for different pathogens (mutational signals)
    and different exponential growth rates, at a fixed present-day population size.

    Parameters:
    pathogen_params (dict): {pathogen name: {'mu', 'k', 'L', 't_max', ...}} with mutation rate,
        number of mutations, genome length, and maximum tMRCA.
    N (int): Effective population size at present.
    beta_vec (list): Growth rates of the population (positive = growth, negative = decline).

    Returns:
    pd.DataFrame: One row per (pathogen, beta).
    """

    data = []

    for name, params in pathogen_params.items():
        t_vec = _time_grid(params['t_max'])
        likelihoods = _likelihoods(params['k'], params['mu'], params['L'], t_vec)

        for beta in beta_vec:
            priors = np.array([coalescent_prior_expN_present(t, N, beta) for t in t_vec])
            posteriors = likelihoods * priors

            data.append({
                'pathogen': name,
                'N': N,
                'beta': beta,
                **_prior_influence_metrics(t_vec, posteriors, likelihoods, abs_value=False),
            })

    return pd.DataFrame(data)


def create_prior_influence_metric_df_bottleneck_pathogen(pathogen_params, N_high, N_low, t_bottleneck_start, t_bottleneck_end_vec):
    """
    Create a DataFrame of prior influence metrics for different pathogens (mutational signals)
    and different bottleneck durations, at fixed bottleneck depth and start time.

    Parameters:
    pathogen_params (dict): {pathogen name: {'mu', 'k', 'L', 't_max', ...}} with mutation rate,
        number of mutations, genome length, and maximum tMRCA.
    N_high (int): Effective population size outside the bottleneck.
    N_low (int): Effective population size during the bottleneck.
    t_bottleneck_start (float): Start time of the bottleneck backwards in time.
    t_bottleneck_end_vec (list): End times of the bottleneck backwards in time.

    Returns:
    pd.DataFrame: One row per (pathogen, t_bottleneck_end).
    """

    data = []

    for name, params in pathogen_params.items():
        t_vec = _time_grid(params['t_max'])
        likelihoods = _likelihoods(params['k'], params['mu'], params['L'], t_vec)

        for t_bottleneck_end in t_bottleneck_end_vec:
            priors = np.array([coalescent_prior_bottleneck(t, N_high, N_low, t_bottleneck_start, t_bottleneck_end) for t in t_vec])
            posteriors = likelihoods * priors

            data.append({
                'pathogen': name,
                'N_high': N_high,
                'N_low': N_low,
                't_bottleneck_start': t_bottleneck_start,
                't_bottleneck_end': t_bottleneck_end,
                **_prior_influence_metrics(t_vec, posteriors, likelihoods, abs_value=False),
            })

    return pd.DataFrame(data)


################ Single mutational signal, varying population model ################

def create_prior_influence_metric_df_pathogen_exp(N_beta_dict, mu, k, L, t_max):
    """
    Create a DataFrame of prior influence metrics for an exponentially growing population at a
    single mutational signal, for several present-day population sizes and growth rates.

    Parameters:
    N_beta_dict (dict): {N_present: list of growth rates beta} to evaluate.
    mu (float): Mutation rate per base pair per year.
    k (int): Number of mutations observed.
    L (int): Genome length.
    t_max (float): Maximum tMRCA to consider.

    Returns:
    pd.DataFrame: One row per (N_present, beta).
    """

    data = []

    t_vec = _time_grid(t_max)
    likelihoods = _likelihoods(k, mu, L, t_vec)

    for N, beta_vec in N_beta_dict.items():
        for beta in beta_vec:
            priors = np.array([coalescent_prior_expN_present(t, N, beta) for t in t_vec])
            posteriors = likelihoods * priors

            data.append({
                'N_present': N,
                'beta': beta,
                **_prior_influence_metrics(t_vec, posteriors, likelihoods, abs_value=False),
            })

    return pd.DataFrame(data)


def create_prior_influence_metric_df_bottleneck(N_high, N_low_values, t_bottleneck_end_vec, t_bottleneck_start, alpha_values, base_mu, base_k, L, t_max):
    """
    Create a DataFrame of prior influence metrics for a constant population size with bottleneck.
    For simplicity, fix the constant population size before and after the bottleneck as well as the start time of the bottleneck.
    Bottleneck depth and duration are varied across the grid.

    Parameters:
    N_high (int): Effective population size outside the bottleneck.
    N_low_values (list): List of effective population sizes during the bottleneck.
    t_bottleneck_end_vec (list): List of end times for the bottleneck backwards in time.
    t_bottleneck_start (int): Start time of the bottleneck backwards in time.
    alpha_values (list): Scaling factors of the mutational signal: mu = base_mu * alpha and
        k = base_k * alpha (at least 1 mutation).
    base_mu (float): Base mutation rate.
    base_k (int): Base number of mutations.
    L (int): Genome length.
    t_max (float): Max time to consider for tMRCA.

    Returns:
    pd.DataFrame: One row per (alpha, N_low, t_bottleneck_end).
    """

    data = []

    t_vec = _time_grid(t_max)

    for alpha in alpha_values:
        mu = base_mu * alpha
        k_mut = max(int(base_k * alpha), 1)  # ensure at least 1 mutation
        likelihoods = _likelihoods(k_mut, mu, L, t_vec)

        for N_low in N_low_values:
            for t_bottleneck_end in t_bottleneck_end_vec:
                priors = np.array([coalescent_prior_bottleneck(t, N_high, N_low, t_bottleneck_start, t_bottleneck_end) for t in t_vec])
                posteriors = likelihoods * priors

                data.append({
                    'N_high': N_high,
                    'N_low': N_low,
                    't_bottleneck_end': t_bottleneck_end,
                    't_bottleneck_start': t_bottleneck_start,
                    'alpha': alpha,
                    **_prior_influence_metrics(t_vec, posteriors, likelihoods, abs_value=False),
                })

    return pd.DataFrame(data)
