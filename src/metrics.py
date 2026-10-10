"""
Metrics that quantify how much the coalescent prior shifts the tMRCA posterior.

Every metric compares two (unnormalized) densities evaluated on the same time grid `t_vec`:
`distr1` is usually the posterior under the coalescent prior and `distr2` the posterior under a
flat prior (i.e. the likelihood). Densities are normalized internally where needed, so the
inputs do not have to integrate to 1.
"""

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.integrate import cumulative_trapezoid


def calculate_wasserstein_dist(t_vec, distr1, distr2):
    """
    Calculate the Wasserstein-1 distance between two distributions.

    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).

    Returns:
    float: Wasserstein distance.
    """

    # Normalize the distributions
    distr1_norm = distr1 / np.trapezoid(distr1, t_vec)
    distr2_norm = distr2 / np.trapezoid(distr2, t_vec)

    # Treat the grid points as support of two discrete distributions weighted by the densities
    w_dist = wasserstein_distance(t_vec, t_vec, u_weights=distr1_norm, v_weights=distr2_norm)

    return w_dist


def _median(t_vec, distr):
    """Median of an (unnormalized) density evaluated on t_vec."""
    distr = distr / np.trapezoid(distr, t_vec)
    cdf = cumulative_trapezoid(distr, t_vec, initial=0)
    # Median is the time point where the CDF crosses 0.5
    return np.interp(0.5, cdf, t_vec)


def calculate_median_shift(t_vec, distr1, distr2, abs_value=True):
    """
    Calculate the median shift (median of distr1 minus median of distr2).

    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    abs_value (bool): Whether to return the absolute value of the shift.

    Returns:
    float: Median shift, in the time unit of t_vec.
    """

    median_shift = _median(t_vec, distr1) - _median(t_vec, distr2)

    if abs_value:
        median_shift = np.abs(median_shift)

    return median_shift


def calculate_rel_median_shift(t_vec, distr1, distr2, abs_value=True):
    """
    Calculate the median shift relative to the median of distr2.

    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    abs_value (bool): Whether to return the absolute value of the shift.

    Returns:
    float: Relative median shift (NaN if the median of distr2 is not positive).
    """

    median1 = _median(t_vec, distr1)
    median2 = _median(t_vec, distr2)

    rel_median_shift = (median1 - median2) / median2 if median2 > 0 else np.nan

    if abs_value:
        rel_median_shift = np.abs(rel_median_shift)

    return rel_median_shift


def calculate_mode_shift(t_vec, distr1, distr2, abs_value=True):
    """
    Calculate the mode shift (mode of distr1 minus mode of distr2).

    Parameters:
    t_vec (np.ndarray): Vector of tMRCA values.
    distr1 (np.ndarray): First distribution (e.g., posterior).
    distr2 (np.ndarray): Second distribution (e.g., likelihood).
    abs_value (bool): Whether to return the absolute value of the shift.

    Returns:
    float: Mode shift, in the time unit of t_vec.
    """

    mode1 = t_vec[np.argmax(distr1)]
    mode2 = t_vec[np.argmax(distr2)]

    mode_shift = mode1 - mode2
    if abs_value:
        mode_shift = np.abs(mode_shift)

    return mode_shift


def calculate_coalescent_information_ratio_at_MAP(N, mu, L, posterior, t_vec, beta=None, N_low=None, t_bottleneck_start=None, t_bottleneck_end=None, population_model='constant', reverse_scale=False):
    """
    Calculate the coalescent information ratio (Omega) at the MAP estimate of the posterior.
    Omega close to 1 means the mutational signal dominates; close to 0 means the prior dominates.

    Parameters:
    N (int): Effective population size for constant population size, N_present for exponential growth and N_high for bottleneck.
    mu (float): Mutation rate per base pair per generation.
    L (int): Length of the genome in base pairs.
    posterior (np.ndarray): Posterior density evaluated on t_vec.
    t_vec (np.ndarray): Vector of tMRCA values.
    beta (float, optional): Growth rate of the population for exponential growth models.
    N_low (int, optional): Population size during the bottleneck for bottleneck models.
    t_bottleneck_start (float, optional): Start time of the bottleneck backwards in time.
    t_bottleneck_end (float, optional): End time of the bottleneck backwards in time.
    population_model (str): Type of population model ('constant', 'exponential', 'bottleneck').
    reverse_scale (bool): If True, return 1 - Omega so that larger values mean more prior influence.

    Returns:
    float: Coalescent information ratio at the MAP estimate.
    """

    # Get the MAP estimate
    t_map_estimate = t_vec[np.argmax(posterior)]

    if population_model == 'constant':
        omega = np.sqrt(1 / (1 + t_map_estimate / (2 * mu * L * N**2)))
    elif population_model == 'exponential':
        if beta is None:
            raise ValueError("beta must be provided for exponential growth models.")
        omega = np.sqrt((2 * mu * L / t_map_estimate) / ((2 * mu * L / t_map_estimate) + beta**2 + 1 / N**2))
    elif population_model == 'bottleneck':
        if N_low is None or t_bottleneck_start is None or t_bottleneck_end is None:
            raise ValueError("N_low, t_bottleneck_start, and t_bottleneck_end must be provided for bottleneck models.")
        # Use the population size that is in effect at the MAP estimate
        N_e = N_low if t_bottleneck_start <= t_map_estimate <= t_bottleneck_end else N
        omega = np.sqrt(1 / (1 + t_map_estimate / (2 * mu * L * N_e**2)))

    if reverse_scale:
        omega = 1 - omega

    return omega
