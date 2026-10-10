# Molecular signal determines the sensitivity of phylogenetic node height estimates to demographic assumptions in Bayesian frameworks

Time-calibrated phylogenies are widely used to reconstruct pathogen evolution and transmission, but in a Bayesian framework node height estimates depend on both molecular information and assumptions about population dynamics. In this study, we investigate when demographic assumptions influence estimates of node heights in dated phylogenies.
We derive analytical posterior distributions for the time to the most recent common ancestor of two sequences under constant population size, exponential growth, and bottlenecks. We show that demographic assumptions can have a large impact on node height estimates, while the influence decreases with increasing molecular signal and population sizes.
We then simulate phylogenies and sequence data and re-estimate node times on fixed topologies using constant-size and skyline coalescent priors. Demographic assumptions have little effect when molecular signal is high, but increasingly affect deep node height estimates as signal declines. The skyline prior reduces these errors under strong population changes.
Finally, we analyse two case studies. For the initial SARS-CoV-2 Alpha variant wave in the UK, a misspecified constant-size prior substantially shifts the estimated root towards the past, whereas a skyline prior recovers more plausible timing. For \textit{Bordetella pertussis} in France, where transmission was halted during the COVID-19 pandemic, a constant-size prior similarly pushes internal node estimates towards older dates.
Overall, our results demonstrate that demographic misspecification can bias the dating of phylogenies, particularly when molecular signal is limited or population size undergoes rapid changes to low levels.

## Part 1: Theoretical analysis
This repository contains the code for the first part of this project. The `theory_plots.ipynb` notebook summarizes all results obtained in this part. Code can be found in the `src` folder.

We consider two sequences sampled at the same time and compute the posterior distribution of their time to the most recent common ancestor (tMRCA) on a time grid. The likelihood is a Poisson distribution on the number of mutations. It is combined with a coalescent prior under constant population size, exponential growth or a bottleneck. We then measure the prior's influence by comparing this posterior to the posterior under a flat prior, using the median shift, relative median shift and Wasserstein-1 distance.

Time is measured in years, backwards from the present, with a generation time of 1. The levels of mutational signal (0.05, 0.5, 5 and 50 mutations per genome per year) are obtained by scaling the mutation rate and mutation count by a factor α. This keeps the tMRCA estimate under a flat prior fixed at 20 years.

### Repository structure
```
├── theory_plots.ipynb                 # Model description and all figures of part 1
├── environment.yaml                   
├── figures/                           # PDF figures written by the notebook
└── src/
    ├── prob_functions.py              # Likelihood and coalescent priors (constant, exponential, bottleneck)
    ├── metrics.py                     # Prior influence metrics (median shift, relative median shift, Wasserstein-1 distance)
    ├── evaluate_prior_impact.py       # DataFrames of metrics over parameter grids
    └── plotting/
        ├── plot_prob_functions.py     # Prior/likelihood/posterior panels
        └── plot_metric_comparison.py  # Summary plots
```

### Setup
Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate pop_size_impact
```
