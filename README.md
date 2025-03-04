# Code Release: Evaluating the Robustness of PCMCI+ for Causal Discovery of Flood Drivers
This repository contains the code used in "Evaluating the Robustness of PCMCI+ for Causal Discovery of Flood Drivers" by Peter Miersch, Wiebke Günther, Jakob Runge, and Jakob Zscheischler (DOI: TO BE ADDED AFTER PUBLICATION). The data is available at Zenodo (DOI: 10.5281/zenodo.14765911)

## Setup
The repository is structured for analysis on a high performance compute cluster with SLURM. All compute-intensive scripts are designed with the `submitit` library to submit jobs to the cluster from withing python. We provide the environment used for our work in `environment.yaml`. This should enable recreating our environment through `conda env create -f environment.yaml`.

The paths are configured in `analysis/conf/io/` where `test.yaml` is for a test run on a single basin and `production.yaml` is for a full run across all 45 basins. Model parameters are controlled with `/analysis/conf/causal_discovery.yaml`. Peak detection is controlled by `/analysis/conf/peak_detection.yaml`, however the discharge peaks used in this study are available alongside the data in tour Zenodo repository.

