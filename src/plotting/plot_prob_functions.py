"""
Plots of the tMRCA prior, likelihood and posterior for different population models, and of the
strength of the mutational signal (posterior SD under a flat prior).
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from src.prob_functions import likelihood_tMRCA_mutations, coalescent_prior, coalescent_prior_expN_present, coalescent_prior_bottleneck
from src.metrics import calculate_coalescent_information_ratio_at_MAP, calculate_wasserstein_dist, calculate_mode_shift, calculate_median_shift
from src.plotting.utils import save_pdf


def _normalize(distr, t_values, scale_mode):
    """
    Scale a density for plotting. "area" normalizes it to integrate to 1 over t_values (a real
    probability density); "max" scales it so its maximum is 1 (easier shape comparison).
    """
    norm = np.trapezoid(distr, t_values) if scale_mode == "area" else np.max(distr)
    return distr / norm if norm > 0 else distr


def _metric_lines(metrics, available_args):
    """
    Evaluate each metric for one panel and format it as "label = value".

    metrics (list of (callable, str, dict)): (metric_fn, label, extra_kwargs) tuples. Each
        metric_fn is called with the entries of available_args whose names match its parameters
        (e.g. t_vec, distr1, distr2, N, mu, L, posterior), updated with extra_kwargs.
    """
    lines = []
    for metric_fn, label, extra_kwargs in metrics:
        try:
            accepted_args = inspect.signature(metric_fn).parameters.keys()
            filtered_args = {k: v for k, v in available_args.items() if k in accepted_args}
            filtered_args.update(extra_kwargs or {})
            lines.append(f"{label} = {metric_fn(**filtered_args):.2f}")
        except Exception:
            # Show the failure in the plot rather than aborting the whole figure
            lines.append(f"{label}: error")
    return lines


def _add_metric_box(ax, lines, x=0.05, ha="left", fontsize=16):
    """Draw the metric lines as a text box in the top corner of ax."""
    if lines:
        ax.text(
            x, 0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment='top',
            horizontalalignment=ha,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.6)
        )


def plot_tMRCA_constN(N_values, alpha_values, base_mu, base_k, t_max, L, time_scale="days", metrics=None, title=None, figsize=None, x_max=None, scale_mode="max", save_path=None, y_range=None, column_labels=None):
    """
    Plot the prior, likelihood, and posterior of tMRCA for a constant population size, in a grid
    of population sizes (rows) and mutational signals (columns).

    Parameters:
    N_values (list): Effective population sizes, one per row.
    alpha_values (list): Scaling factors of the mutational signal, one per column:
        mu = base_mu * alpha and k = base_k * alpha (at least 1 mutation).
    base_mu (float): Base mutation rate per base pair per time unit.
    base_k (int): Base number of mutations.
    t_max (int): Maximum tMRCA used for the time grid (and thus for the metric calculation).
    L (int): Genome length.
    time_scale (str): Time unit shown in the x-axis label.
    metrics (list of (callable, str, dict), optional): (metric_fn, label, extra_kwargs) tuples to
        compute and annotate in each panel. The metric compares the posterior (distr1) to the
        likelihood (distr2). No annotation if not set.
    title (str, optional): Figure suptitle. Defaults to a generic title if not set.
    figsize (tuple, optional): Figure size. Defaults to (20, 12).
    x_max (float, optional): Upper limit of the x-axis shown in the plot, independent of t_max
        (which sets the range used for metric calculation). Defaults to t_max if not set.
    scale_mode (str): How to scale the plotted curves. "max" (default) scales each curve by its
        own maximum, so all curves peak at 1 (easier shape comparison, not a real probability
        density). "area" normalizes each curve to integrate to 1 over t_values, giving the actual
        probability densities.
    save_path (str, optional): If set, the figure is saved there as PDF.
    y_range (tuple, optional): (y_min, y_max) applied to every subplot. Defaults to each
        subplot's automatic y-range if not set.
    column_labels (list, optional): Custom column titles, one per entry in alpha_values (e.g. the
        mutational signal each alpha corresponds to, like "0.05 mutations per genome per year").
        Defaults to "α = {alpha}" if not set.
    """

    if column_labels is not None and len(column_labels) != len(alpha_values):
        raise ValueError("column_labels must have the same length as alpha_values.")
    if scale_mode not in ("max", "area"):
        raise ValueError(f"scale_mode must be 'max' or 'area', got {scale_mode!r}")

    t_values = np.linspace(0, t_max, t_max * 10)
    x_max = t_max if x_max is None else x_max

    colors = ["#a6444f", "#397398", "#80557e"]  # Prior, Likelihood, Posterior

    fig, axs = plt.subplots(len(N_values), len(alpha_values), figsize=(20, 12) if figsize is None else figsize, squeeze=False)

    for row, N in enumerate(N_values):
        for col, alpha in enumerate(alpha_values):
            mu = base_mu * alpha
            k_mut = max(int(base_k * alpha), 1)

            likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
            priors = np.array([coalescent_prior(t, N) for t in t_values])
            posteriors = likelihoods * priors

            likelihoods = _normalize(likelihoods, t_values, scale_mode)
            priors = _normalize(priors, t_values, scale_mode)
            posteriors = _normalize(posteriors, t_values, scale_mode)

            ax = axs[row, col]
            line_prior, = ax.plot(t_values, priors, linestyle="--", color=colors[0], label="Prior", linewidth=4)
            line_likelihood, = ax.plot(t_values, likelihoods, linestyle=":", color=colors[1], label="Posterior under flat prior", linewidth=4)
            line_posterior, = ax.plot(t_values, posteriors, linewidth=4, color=colors[2], label="Posterior under coalescent prior")
            ax.grid(True)
            ax.set_xlim(0, x_max)
            if y_range is not None:
                ax.set_ylim(*y_range)

            ax.set_xlabel(f"tMRCA [{time_scale}]", fontsize=22)
            y_label = "Rel. prob. (scaled by max.)" if scale_mode == "max" else "Probability density"
            ax.set_ylabel(y_label, fontsize=22)
            # Row label (N) left of the first column, column title above the first row
            if col == 0:
                ax.annotate(
                    f"N = {int(N)}",
                    xy=(-0.28, 0.5), xycoords="axes fraction",
                    rotation=90, ha="center", va="center", fontsize=24,
                )
            if row == 0:
                col_title = f"α = {alpha}" if column_labels is None else column_labels[col]
                ax.set_title(col_title, fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=18)

            if metrics:
                available_args = {
                    "t_vec": t_values,
                    "distr1": posteriors,
                    "distr2": likelihoods,
                    "N": N,
                    "mu": mu,
                    "L": L,
                    "posterior": posteriors,
                }
                _add_metric_box(ax, _metric_lines(metrics, available_args))

    if title is None:
        title = "Prior, Likelihood, and Posterior of tMRCA at varying N and α"
    fig.suptitle(title, fontsize=28)
    fig.legend(
        [line_prior, line_likelihood, line_posterior],
        ["Prior", "Posterior under flat prior", "Posterior under coalescent prior"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=fig.transFigure,
        fontsize=18,
        ncol=3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.2, w_pad=2.0)
    if save_path is not None:
        save_pdf(save_path)
    plt.show()


def plot_tMRCA_expN_present(mu, L, k_mut, N_present, beta_vec, max_tmrca, N=None, plot_comparison_to_const=True, time_scale="days", x_max=None, scale_mode="max", save_path=None, metrics=None, y_range=None, figsize=None):
    """
    Plot the prior, likelihood, and posterior of tMRCA for different exponential growth rates (β),
    one panel per growth rate. Population size is N(t) = N_present * exp(-β t) backwards in time.

    Parameters:
    mu (float): Mutation rate per base per generation.
    L (int): Length of the genome in base pairs.
    k_mut (int): Number of mutations observed.
    N_present (int): Effective population size at present.
    beta_vec (list): List of exponential growth rates.
    max_tmrca (int): Maximum tMRCA used for the time grid (and thus for the metric calculation).
    N (int, optional): Constant population size for the comparison prior/posterior. Only used
        if plot_comparison_to_const is True.
    plot_comparison_to_const (bool): If True, also plot prior and posterior under a constant
        population size N.
    time_scale (str): Time unit shown in the x-axis label.
    x_max (float, optional): Upper limit of the x-axis shown in the plot, independent of max_tmrca
        (which sets the range used for metric calculation). Defaults to max_tmrca if not set.
    scale_mode (str): How to scale the plotted curves. "max" (default) scales each curve by its
        own maximum, so all curves peak at 1 (easier shape comparison, not a real probability
        density). "area" normalizes each curve to integrate to 1 over t_values, giving the actual
        probability densities.
    save_path (str, optional): If set, the figure is saved there as PDF.
    metrics (list of (callable, str, dict), optional): List of (metric_fn, label, extra_kwargs)
        tuples to compute and annotate in each panel, same convention as in plot_tMRCA_constN.
        Defaults to Wasserstein distance, reverse coalescent information ratio, and median shift
        if not set.
    y_range (tuple, optional): (y_min, y_max) applied to every subplot. Defaults to each
        subplot's automatic y-range if not set.
    figsize (tuple, optional): Figure size. Defaults to (20, 5).
    """

    if scale_mode not in ("max", "area"):
        raise ValueError(f"scale_mode must be 'max' or 'area', got {scale_mode!r}")

    t_values = np.linspace(0, max_tmrca, max_tmrca)
    x_max = max_tmrca if x_max is None else x_max

    colors = [
        "#b7b5b5",  # prior constant
        "#397398",  # likelihood
        "#6c6c6c",  # posterior constant
        "#a6444f",  # prior expN
        "#80557e",  # posterior expN
    ]

    if metrics is None:
        metrics = [
            (calculate_wasserstein_dist, "W. dist.", {}),
            (calculate_coalescent_information_ratio_at_MAP, "1 - Ω", {'population_model': 'exponential', 'reverse_scale': True}),
            (calculate_median_shift, "Median shift", {'abs_value': False}),
        ]

    fig, axs = plt.subplots(1, len(beta_vec), figsize=(20, 5) if figsize is None else figsize, squeeze=False)
    axs = axs[0]

    for col, beta in enumerate(beta_vec):
        likelihoods = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
        priors_expN = np.array([coalescent_prior_expN_present(t, N_present, beta) for t in t_values])
        posteriors_expN = likelihoods * priors_expN

        likelihoods = _normalize(likelihoods, t_values, scale_mode)
        priors_expN = _normalize(priors_expN, t_values, scale_mode)
        posteriors_expN = _normalize(posteriors_expN, t_values, scale_mode)

        ax = axs[col]
        legend_handles = []
        legend_labels = []
        if plot_comparison_to_const:
            priors = np.array([coalescent_prior(t, N) for t in t_values])
            posteriors = _normalize(likelihoods * priors, t_values, scale_mode)
            priors = _normalize(priors, t_values, scale_mode)

            line, = ax.plot(t_values, priors, linestyle="--", color=colors[0], alpha=0.8, linewidth=4)
            legend_handles.append(line); legend_labels.append("Prior (const N)")
            line, = ax.plot(t_values, posteriors, linewidth=4, color=colors[2], alpha=0.8)
            legend_handles.append(line); legend_labels.append("Posterior (const N)")

        line, = ax.plot(t_values, likelihoods, linestyle=":", color=colors[1], alpha=0.8, linewidth=4)
        legend_handles.append(line); legend_labels.append("Posterior under flat prior")
        line, = ax.plot(t_values, priors_expN, linestyle="-.", color=colors[3], alpha=0.8, linewidth=4)
        legend_handles.append(line); legend_labels.append("Exp. growth prior")
        line, = ax.plot(t_values, posteriors_expN, linewidth=4, color=colors[4], alpha=0.8)
        legend_handles.append(line); legend_labels.append("Posterior under coalescent prior")

        ax.set_title(f"β = {beta}", fontsize=24)
        ax.set_xlabel(f"tMRCA [{time_scale}]", fontsize=22)
        ax.set_xlim(0, x_max)
        if y_range is not None:
            ax.set_ylim(*y_range)
        if col == 0:
            y_label = "Rel. prob. (scaled by max.)" if scale_mode == "max" else "Probability density"
            ax.set_ylabel(y_label, fontsize=22)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=18)

        # Metrics compare the posterior under the exponential growth prior to the likelihood (flat prior)
        available_args = {
            "t_vec": t_values,
            "distr1": posteriors_expN,
            "distr2": likelihoods,
            "N": N_present,
            "mu": mu,
            "L": L,
            "beta": beta,
            "posterior": posteriors_expN,
        }
        _add_metric_box(ax, _metric_lines(metrics, available_args))

    fig.legend(
        legend_handles, legend_labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=fig.transFigure,
        fontsize=18,
        ncol=len(legend_labels),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=1.2, w_pad=2.0)
    if save_path is not None:
        save_pdf(save_path)
    plt.show()


def plot_tMRCA_bottleneck(mu, L, k_mut, N_high, N_low_vec, t_bottleneck_start, t_bottleneck_end_vec, t_max, time_scale="days", figsize=None, x_max=None, scale_mode="max", save_path=None, max_cols=None, title=None, tile_labels=None, metrics=None, y_range=None):
    """
    Plot the prior, likelihood, and posterior of tMRCA for varying bottleneck depths and durations.

    Parameters:
    mu (float): Mutation rate per base pair per generation.
    L (int): Length of the genome in base pairs.
    k_mut (int): Number of mutations observed.
    N_high (int): Population size outside the bottleneck.
    N_low_vec (list): List of population sizes during the bottleneck.
    t_bottleneck_start (int): Start time of the bottleneck backwards in time.
    t_bottleneck_end_vec (list): List of end times for the bottleneck backwards in time.
    t_max (int): Maximum tMRCA to consider.
    time_scale (str): Time unit shown in the x-axis label.
    figsize (tuple, optional): Figure size. Defaults to 5 x 4.5 per panel.
    x_max (float, optional): Upper limit of the x-axis shown in the plot, independent of t_max
        (which sets the range used for metric calculation). Defaults to t_max if not set.
    scale_mode (str): How to scale the plotted curves. "max" (default) scales each curve by its
        own maximum, so all curves peak at 1 (easier shape comparison, not a real probability
        density). "area" normalizes each curve to integrate to 1 over t_values, giving the actual
        probability densities.
    save_path (str, optional): If set, the figure is saved there as PDF.
    max_cols (int, optional): Maximum number of subplots per row. If set, all (N_low,
        t_bottleneck_end) combinations are flattened and wrapped into a grid with at most
        max_cols columns per row, instead of the default N_low_vec x t_bottleneck_end_vec grid.
    title (str, optional): Figure suptitle. Pass "" for no title. Defaults to
        "tMRCA Prior, Likelihood, and Posterior\\nat varying bottleneck duration" if not set.
    tile_labels (list of str, optional): Custom per-subplot titles, overriding the default
        "t_end = ..." title. Without max_cols: one label per entry in t_bottleneck_end_vec
        (used as the shared column title). With max_cols: one label per (N_low,
        t_bottleneck_end) panel, in the same flattened order as the wrapped grid.
    metrics (list of (callable, str, dict), optional): List of (metric_fn, label, extra_kwargs)
        tuples to compute and annotate in each panel, same convention as in plot_tMRCA_constN.
        Defaults to Wasserstein distance, reverse coalescent information ratio, mode shift, and
        median shift if not set.
    y_range (tuple, optional): (y_min, y_max) applied to every subplot. Defaults to each
        subplot's automatic y-range if not set.
    """
    colors = [
        "#397398",  # likelihood
        "#a6444f",  # prior bottleneck
        "#80557e",  # posterior bottleneck
    ]

    if scale_mode not in ("max", "area"):
        raise ValueError(f"scale_mode must be 'max' or 'area', got {scale_mode!r}")

    t_values = np.linspace(0, t_max, t_max)
    x_max = t_max if x_max is None else x_max
    y_label = "Rel. prob. (scaled by max.)" if scale_mode == "max" else "Probability density"

    # One panel per (N_low, t_bottleneck_end) combination
    panels = [(N_low, t_bottleneck_end) for N_low in N_low_vec for t_bottleneck_end in t_bottleneck_end_vec]

    # Grid layout: either wrapped to max_cols columns, or rows = N_low and columns = t_bottleneck_end
    ncols = max_cols if max_cols is not None else len(t_bottleneck_end_vec)
    nrows = -(-len(panels) // ncols)  # ceil division

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows) if figsize is None else figsize, squeeze=False)

    multiple_N_low = len(N_low_vec) > 1

    # The likelihood does not depend on the bottleneck, so compute it once
    likelihoods_raw = np.array([likelihood_tMRCA_mutations(k_mut, mu, t, L) for t in t_values])
    likelihoods = _normalize(likelihoods_raw, t_values, scale_mode)

    for idx, (N_low, t_bottleneck_end) in enumerate(panels):
        row, col = divmod(idx, ncols)

        priors_bottleneck = np.array([
            coalescent_prior_bottleneck(t, N_high, N_low, t_bottleneck_start, t_bottleneck_end)
            for t in t_values
        ])
        posteriors_bottleneck = _normalize(likelihoods_raw * priors_bottleneck, t_values, scale_mode)
        priors_bottleneck = _normalize(priors_bottleneck, t_values, scale_mode)

        ax = axs[row, col]
        line_likelihood, = ax.plot(t_values, likelihoods, linestyle=":", color=colors[0], label="Posterior under flat prior", linewidth=3)
        line_prior, = ax.plot(t_values, priors_bottleneck, linestyle="--", color=colors[1], label="Bottleneck prior", linewidth=3)
        line_posterior, = ax.plot(t_values, posteriors_bottleneck, color=colors[2], label="Posterior under coalescent prior", linewidth=3)
        ax.grid(True)
        ax.set_xlim(0, x_max)
        if y_range is not None:
            ax.set_ylim(*y_range)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel(f"tMRCA [{time_scale}]", fontsize=14)

        if max_cols is None:
            # Regular grid: N_low in the y-label of the first column, t_end as column title
            if col == 0:
                ax.set_ylabel(f"N_low = {N_low}\n{y_label}", fontsize=14)
            if row == 0:
                panel_title = f"t_end = {t_bottleneck_end}" if tile_labels is None else tile_labels[col]
                ax.set_title(panel_title, fontsize=15)
        else:
            # Wrapped grid: every panel gets its own title
            ax.set_ylabel(y_label, fontsize=14)
            if tile_labels is not None:
                panel_title = tile_labels[idx]
            else:
                panel_title = f"t_end = {t_bottleneck_end}"
                if multiple_N_low:
                    panel_title = f"N_low = {N_low}, {panel_title}"
            ax.set_title(panel_title, fontsize=15)

        panel_metrics = metrics
        if panel_metrics is None:
            panel_metrics = [
                (calculate_wasserstein_dist, "W. dist.", {}),
                (calculate_coalescent_information_ratio_at_MAP, "1 - Ω", {'population_model': 'bottleneck', 'reverse_scale': True}),
                (calculate_mode_shift, "Mode shift", {'abs_value': False}),
                (calculate_median_shift, "Median shift", {'abs_value': False}),
            ]

        # Metrics compare the posterior under the bottleneck prior to the likelihood (flat prior)
        available_args = {
            "t_vec": t_values,
            "distr1": posteriors_bottleneck,
            "distr2": likelihoods,
            "N": N_high,
            "mu": mu,
            "L": L,
            "N_low": N_low,
            "t_bottleneck_start": t_bottleneck_start,
            "t_bottleneck_end": t_bottleneck_end,
            "posterior": posteriors_bottleneck,
        }
        _add_metric_box(ax, _metric_lines(panel_metrics, available_args), x=0.97, ha="right", fontsize=13)

    # Hide any unused trailing axes when panels don't fill the grid evenly
    for idx in range(len(panels), nrows * ncols):
        row, col = divmod(idx, ncols)
        axs[row, col].axis("off")

    fig.legend(
        handles=[line_likelihood, line_prior, line_posterior],
        loc='upper right',
        bbox_to_anchor=(1, 1.0),
        fontsize=15,
        ncol=3,
    )
    if title is None:
        title = "tMRCA Prior, Likelihood, and Posterior\nat varying bottleneck duration"
    if title:
        fig.suptitle(title, fontsize=16)
        rect_top = 0.92
    else:
        rect_top = 0.94
    plt.tight_layout(rect=[0, 0, 1, rect_top])
    if save_path is not None:
        save_pdf(save_path)
    plt.show()


def _std_mut(T, mu_genome):
    """
    SD of the tMRCA posterior under a flat prior. With k = 2 * T * mu' - 1 mutations (so that the
    posterior mean (k + 1) / (2 mu') equals T), Var = (k + 1) / (2 mu')^2.
    """
    k = 2 * T * mu_genome - 1
    return np.sqrt(k + 1) / (2 * mu_genome)


def plot_mutational_signal_std(base_mu, L, base_k, alpha_values, pathogens=None, mu_range=(1e-2, 1e3), t_range=(10**-1.5, 1e3), std_levels=None, signal_colors=None, figsize=(8, 6.5), title=None, save_path=None):
    """
    Contour map of the std (in years) of the tMRCA posterior with a flat prior over the (mu', T)
    plane, with mu' = mutations per genome per year and T = tree height in years. Var = (k + 1) / (2 mu')^2 with
    k = 2 * T * mu'.

    Parameters:
    base_mu (float): Base per-site mutation rate per year.
    L (int): Genome length.
    base_k (float): Base number of mutations. Each alpha scenario uses k = base_k * alpha and
        mu' = alpha * base_mu * L, placed at tree height T = (k + 1) / (2 * mu'), consistent with
        how pathogen points are placed.
    alpha_values (list): Scaling factors of the mutation rate and of base_k.
    pathogens (dict, optional): {label: (T, mutations_per_genome_per_year)} or
        {label: ((T_min, T_max), mutations_per_genome_per_year)} of real pathogens, where T (or
        T_min/T_max) is that pathogen's tree height in years and mutations_per_genome_per_year is
        its mu'. c is assumed to be 2 throughout (two lineages), so std is read directly off the
        background shading. A single T is drawn as a point; a (T_min, T_max) tuple is drawn as a
        grey ellipse spanning that tree-height range at the given mu', to reflect uncertainty in
        the tMRCA. Each pathogen is labelled directly next to its point/ellipse rather than in a
        legend. A warning is printed (not raised) if a pathogen falls outside mu_range or t_range.
    mu_range, t_range (tuple): Axis limits for mu' and T.
    std_levels (array, optional): Contour levels of the std in years. Defaults to 0.001 to 1000.
    signal_colors (list, optional): One color per entry in alpha_values for the base scenario
        points. Defaults to the palette used in plot_pathogen_error_summary.
    figsize (tuple): Figure size.
    title (str, optional): Axes title. No title if not set.
    save_path (str, optional): If set, the figure is saved there as PDF.
    """
    std_levels = np.logspace(-3, 3, 25) if std_levels is None else std_levels
    signal_colors = ["#a6444f", "#80557e", "#7394c2", "#397398"] if signal_colors is None else signal_colors
    if len(signal_colors) < len(alpha_values):
        raise ValueError("signal_colors must have at least as many entries as alpha_values.")

    # Background: posterior SD on a log-spaced (mu', T) grid
    mu_grid = np.logspace(np.log10(mu_range[0]), np.log10(mu_range[1]), 300)
    t_grid = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), 300)
    MU, T = np.meshgrid(mu_grid, t_grid)
    S = _std_mut(T, MU)

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(MU, T, S, levels=std_levels, norm=LogNorm(), cmap="Greys", alpha=0.6)
    ax.contour(MU, T, S, levels=std_levels[::2], colors="k", linewidths=0.5)
    cbar = fig.colorbar(cf)
    cbar.set_label("SD of tree height posterior\nw. flat prior [years]", fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    # Colored points for the simulated mutational signal scenarios
    mu_genome = base_mu * L
    signal_handles = []
    for a, color in zip(alpha_values, signal_colors):
        k_a = base_k * a
        mu_a = a * mu_genome
        T_a = (k_a + 1) / (2 * mu_a)
        signal_handles.append(ax.scatter(mu_a, T_a, c=color, edgecolor="k", s=150, zorder=3, label=f"{mu_a:g}"))

    # Grey markers (or ellipses for a T range) for real pathogens, labelled in place
    if pathogens:
        markers = ["s", "^", "D", "v", "P", "X"]
        log_mu_span = np.log10(mu_range[1]) - np.log10(mu_range[0])
        for i, (label, (T_path, mu_path)) in enumerate(pathogens.items()):
            if isinstance(T_path, (tuple, list)):
                # Ellipse in log space spanning [T_min, T_max] vertically
                T_min, T_max = T_path
                log_mu, log_T_mid = np.log10(mu_path), (np.log10(T_min) + np.log10(T_max)) / 2
                semi_y = (np.log10(T_max) - np.log10(T_min)) / 2
                semi_x = 0.04 * log_mu_span
                theta = np.linspace(0, 2 * np.pi, 100)
                ax.fill(10 ** (log_mu + semi_x * np.cos(theta)), 10 ** (log_T_mid + semi_y * np.sin(theta)),
                        facecolor="#7a7a7a", edgecolor="k", alpha=0.5, linewidth=1, zorder=3)
                label_pos = (10 ** (log_mu + semi_x), 10 ** log_T_mid)
                T_check = (T_min, T_max)
            else:
                ax.scatter(mu_path, T_path, marker=markers[i % len(markers)], c="#7a7a7a", edgecolor="k", s=130, zorder=3)
                label_pos = (mu_path, T_path)
                T_check = (T_path, T_path)
            ax.annotate(label, label_pos, xytext=(10, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=14, color="#333333", style="italic", clip_on=True)
            if not (mu_range[0] <= mu_path <= mu_range[1] and t_range[0] <= T_check[0] and T_check[1] <= t_range[1]):
                print(f"Warning: {label} at (mu'={mu_path:g}, T={T_check}) falls outside the plotted mu_range={mu_range} or t_range={t_range} and may not be fully visible.")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(mu_range)
    ax.set_ylim(t_range)
    ax.set_xlabel("Mutations per genome per year", fontsize=22)
    ax.set_ylabel("Expected tree height [years]", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    if title is not None:
        ax.set_title(title, fontsize=24)
    ax.legend(handles=signal_handles, title="Mutations per genome\nper year", loc="lower left",
              fontsize=13, title_fontsize=14, frameon=True, facecolor="#ececec", edgecolor="#7a7a7a", framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        save_pdf(save_path)
    plt.show()


def plot_mutational_signal_std_by_rate(base_mu, L, base_k, alpha_values, signal_colors=None, n_line_points=200, n_small_dots=20, log_y=True, figsize=(8, 6.5), title=None, save_path=None):
    """
    Std (in years) of the mutational signal against mutations per genome per year. Var = (k + 1) / (2 mu')^2
    with mu' = mutations per genome per year and k = 2 * T * mu'.

    Parameters:
    base_mu (float): Base per-site mutation rate per year.
    L (int): Genome length.
    base_k (float): Base number of mutations; sets the base tMRCA T = (base_k + 1) / (2 * base_mu * L).
    alpha_values (list): Scaling factors of the mutation rate. Each gives one base scenario at
        mu' = alpha * base_mu * L and the shared T. The scenarios are connected by a line.
    signal_colors (list, optional): One color per entry in alpha_values. Defaults to the palette
        used in plot_pathogen_error_summary.
    n_line_points (int): Number of log-spaced points between the lowest and highest rate used to
        draw the smooth line.
    n_small_dots (int): Number of log-spaced small grey dots drawn on the line between the
        highlighted scenarios, to show that the line is built from more data points.
    log_y (bool): If True (default), the y-axis is logarithmic. If False, it is linear.
    figsize (tuple): Figure size.
    title (str, optional): Axes title. No title if not set.
    save_path (str, optional): If set, the figure is saved there as PDF.
    """
    signal_colors = ["#a6444f", "#80557e", "#7394c2", "#397398"] if signal_colors is None else signal_colors
    if len(signal_colors) < len(alpha_values):
        raise ValueError("signal_colors must have at least as many entries as alpha_values.")

    # All scenarios share the same tree height; only the mutation rate changes
    mu_genome = base_mu * L
    T_base = (base_k + 1) / (2 * mu_genome)
    mu_vals = np.array(sorted(a * mu_genome for a in alpha_values))
    std_vals = _std_mut(T_base, mu_vals)

    fig, ax = plt.subplots(figsize=figsize)

    # Smooth grey line with small dots, then the highlighted scenarios on top
    mu_line = np.logspace(np.log10(mu_vals[0]), np.log10(mu_vals[-1]), n_line_points)
    ax.plot(mu_line, _std_mut(T_base, mu_line), color="#7a7a7a", linewidth=2.5, zorder=2)
    mu_dots = np.logspace(np.log10(mu_vals[0]), np.log10(mu_vals[-1]), n_small_dots)
    ax.scatter(mu_dots, _std_mut(T_base, mu_dots), c="#7a7a7a", s=22, zorder=2)
    signal_handles = []
    for mu_val, std_val, color in zip(mu_vals, std_vals, signal_colors):
        signal_handles.append(ax.scatter(mu_val, std_val, c=color, edgecolor="k", s=150, zorder=3, label=f"{mu_val:g}"))

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Mutations per genome per year", fontsize=22)
    ax.set_ylabel("SD of tree height posterior\nw. flat prior [years]", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.grid(True, which="both", ls="--", linewidth=0.5)
    if title is not None:
        ax.set_title(title, fontsize=24)
    ax.legend(handles=signal_handles, title="Mutations per genome\nper year", loc="upper right",
              fontsize=16, title_fontsize=17, frameon=True, facecolor="#ececec", edgecolor="#7a7a7a", framealpha=0.9)
    fig.tight_layout()

    if save_path is not None:
        save_pdf(save_path)
    plt.show()
