"""
Summary plots of prior influence metrics (from src.evaluate_prior_impact) across population models.
"""

import matplotlib.pyplot as plt
from src.plotting.utils import save_pdf


def _plot_metric_summary_exp(ax, df_exp, metric, colors, y_label, legend_title=None):
    """Panel: metric over growth rate β, one line per present-day population size."""
    exp_df = df_exp.copy()
    Ns_exp = sorted(exp_df["N_present"].unique())

    for i, N in enumerate(Ns_exp):
        df_N = exp_df[exp_df["N_present"] == N].sort_values("beta")
        ax.plot(
            df_N["beta"],
            df_N[metric],
            marker="o",
            label=f"N = {N}",
            color=colors[i % len(colors)]
        )

    ax.set_xscale("log")
    ax.set_xlabel("Growth rate β (log scale)", fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.set_title("Exponential growth", fontsize=24)
    ax.grid(True, which='both', ls='--', linewidth=0.5)
    ax.legend(title="N_present" if legend_title is None else legend_title, fontsize=17, title_fontsize=18)
    ax.axhline(0, color="black", linestyle="dotted", linewidth=1)


def _plot_metric_summary_bottleneck(ax, df_bottleneck, metric, colors, y_label, legend_title=None):
    """Panel: metric over bottleneck duration, one line per bottleneck population size N_low."""
    bottleneck_df = df_bottleneck.copy()
    bottleneck_df["bottleneck_duration"] = bottleneck_df["t_bottleneck_end"] - bottleneck_df["t_bottleneck_start"]
    N_lows = sorted(bottleneck_df["N_low"].unique())

    for i, N_low in enumerate(N_lows):
        df_Nlow = bottleneck_df[bottleneck_df["N_low"] == N_low].sort_values("t_bottleneck_end")
        ax.plot(
            df_Nlow["bottleneck_duration"],
            df_Nlow[metric],
            marker="o",
            label=f"N_low = {N_low}",
            color=colors[i % len(colors)]
        )

    ax.set_xlabel("Bottleneck duration [years]", fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    ax.set_title("Population bottleneck", fontsize=24)
    ax.grid(True, which='both', ls='--', linewidth=0.5)
    ax.legend(title="N_low" if legend_title is None else legend_title, fontsize=17, title_fontsize=18)
    ax.axhline(0, color="black", linestyle="dotted", linewidth=1)


def plot_metric_summary(title, df_exp=None, df_bottleneck=None, metric="median_shift", y_label=None, save_path=None, legend_title=None):
    """
    Plot one prior influence metric for a single mutational signal across population models,
    one panel per DataFrame provided.

    Parameters:
    title (str): Currently not drawn (the suptitle is disabled for the paper figures).
    df_exp (pd.DataFrame, optional): Exponential growth results from
        create_prior_influence_metric_df_pathogen_exp (columns N_present, beta, metrics).
    df_bottleneck (pd.DataFrame, optional): Bottleneck results from
        create_prior_influence_metric_df_bottleneck.
    metric (str): Metric column to plot.
    y_label (str, optional): y-axis label. Defaults to "{metric} [years]".
    save_path (str, optional): If set, the figure is saved there as PDF.
    legend_title (str, optional): Overrides the default legend title of each panel.
    """
    # Define custom colors: shades of purple/pink, starting from the same purple used in
    # plot_pathogen_error_summary (so the matching line has the same color), getting lighter
    colors = [
        "#80557e", "#d991b4", "#f2c4d8", "#fbe6f0",
    ]

    # Exponential panel: purple needs to be the 2nd line, so add a darker shade before it
    # and drop the lightest shade
    exp_colors = [
        "#4b3866", "#80557e", "#d991b4", "#f2c4d8",
    ]

    y_label = f"{metric} [years]" if y_label is None else y_label

    # Only include tiles for the DataFrames that were provided
    panels = []
    if df_exp is not None:
        panels.append((df_exp, _plot_metric_summary_exp, exp_colors))
    if df_bottleneck is not None:
        panels.append((df_bottleneck, _plot_metric_summary_bottleneck, colors))

    if not panels:
        raise ValueError("At least one of df_exp, df_bottleneck must be provided.")

    # Create subplots
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (df, plot_fn, panel_colors) in zip(axes, panels):
        plot_fn(ax, df, metric, panel_colors, y_label, legend_title)
        ax.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        save_pdf(save_path)
    plt.show()

def plot_pathogen_error_summary(title, df_const=None, df_growth=None, df_bottleneck=None, error_col="median_shift", y_label=None, save_path=None, legend_position="in"):
    """
    Plot one prior influence metric for all pathogens (mutational signals) across population
    models, one panel per DataFrame provided and one line per pathogen. Panels share the y-axis.

    Parameters:
    title (str): Figure suptitle.
    df_const (pd.DataFrame, optional): Result of create_prior_influence_metric_df_const_pathogen.
    df_growth (pd.DataFrame, optional): Result of create_prior_influence_metric_df_exp_pathogen.
    df_bottleneck (pd.DataFrame, optional): Result of
        create_prior_influence_metric_df_bottleneck_pathogen.
    error_col (str): Metric column to plot.
    y_label (str, optional): y-axis label. Defaults to error_col.
    save_path (str, optional): If set, the figure is saved there as PDF.
    legend_position (str): Where to place the legend. "in" (default) puts a legend in every
        subplot. "below" puts a single shared legend below the whole figure. "none" hides the
        legend entirely.
    """
    if legend_position not in ("in", "below", "none"):
        raise ValueError(f"legend_position must be 'in', 'below', or 'none', got {legend_position!r}")

    colors = [
        "#a6444f", "#80557e", "#7394c2", "#397398",
    ]

    y_label = error_col if y_label is None else y_label

    # Configs for each subplot, only included if the corresponding df is provided
    plot_configs = []

    if df_const is not None:
        plot_configs.append({
            "df": df_const,
            "x_col": "N",
            "x_label": "Population size N (log scale)",
            "title": "Constant population size",
            "xscale": "log",
        })

    if df_growth is not None:
        plot_configs.append({
            "df": df_growth,
            "x_col": "beta",
            "x_label": "Growth rate β (log scale)",
            "title": "Exponential growth",
            "xscale": "log",
        })

    if df_bottleneck is not None:
        plot_configs.append({
            "df": df_bottleneck.assign(bottleneck_duration=lambda d: d["t_bottleneck_end"] - d["t_bottleneck_start"]),
            "x_col": "bottleneck_duration",
            "x_label": "Bottleneck duration [years]",
            "title": "Population bottleneck",
            "xscale": "linear",
        })

    if not plot_configs:
        raise ValueError("At least one of df_const, df_growth, df_bottleneck must be provided.")

    pathogens = sorted(plot_configs[0]["df"]["pathogen"].unique())

    # Create subplots with shared y-axis
    fig, axes = plt.subplots(1, len(plot_configs), figsize=(7.5 * len(plot_configs), 6), sharey=True)
    if len(plot_configs) == 1:
        axes = [axes]

    handles, labels = [], []
    for ax, config in zip(axes, plot_configs):
        df = config["df"]
        x_col = config["x_col"]

        for i, pathogen in enumerate(pathogens):
            df_p = df[df["pathogen"] == pathogen].sort_values(x_col)
            line, = ax.plot(df_p[x_col], df_p[error_col], marker="o", color=colors[i % len(colors)], label=pathogen)
            if ax is axes[0]:
                handles.append(line)
                labels.append(pathogen)

        ax.set_xscale(config["xscale"])
        ax.set_xlabel(config["x_label"], fontsize=22)
        ax.set_title(config["title"], fontsize=24)
        ax.grid(True, which='both', ls='--', linewidth=0.5)
        ax.axhline(0, color="black", linestyle="dotted", linewidth=1)
        ax.tick_params(labelleft=True)  # ensure y-axis ticks are shown for each
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_ylabel(y_label, fontsize=22)
        if legend_position == "in":
            ax.legend(title="Mutations per genome per year", fontsize=17, title_fontsize=18)

    if legend_position == "below":
        fig.legend(
            handles, labels, title="Mutations per genome per year",
            loc="upper center", bbox_to_anchor=(0.5, 0.0),
            ncol=1, fontsize=18, title_fontsize=20
        )

    fig.suptitle(title, fontsize=28)
    if save_path is not None:
        save_pdf(save_path)
    plt.show()
