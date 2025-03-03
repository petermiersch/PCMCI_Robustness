import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import submitit
import time

from inout import read_single_basin
from preprocessing import moving_average
from linear_operator.utils.errors import NotPSDError

from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite import data_processing as pp

from discovery_result_dataclass import discovery_result

from causal_discovery_tools import get_link_assumptions, split_list_into_subsets, mask_builder, run_pcmciplus, run_pcmciplus_full_basin, run_pcmciplus_obs

@hydra.main(version_base=None, config_path="conf", config_name="causal_discovery")
def main(conf: DictConfig) -> None:
    # link assumptions used for all basins
    link_assumptions = get_link_assumptions()
    # iterate through basins by their id
    for id in conf.io.ids:
        # read timeseries data
        basin = read_single_basin(directory=os.path.join(conf.io.data,'resampled'), id=id)
        # create temperature as 7 day moving average
        basin["tavg7"] = moving_average(basin["tavg"], 7)
        basin = basin[["pre", "tavg7", "snow", "SM", "Q"]]
        # read the peak indices created in previous step
        peak_indices = []
        with open(
            os.path.join(os.path.join(conf.io.data,'resampled'), f"{id}_peak_indices_USWRC.txt"), "r"
        ) as filehandle:
            peak_indices = [
                current_place.rstrip() for current_place in filehandle.readlines()
            ]
            # convert the read list to integers (instead of strings)
            peak_indices = list(map(int, peak_indices))
        # number of peaks (=samples) available for this catchment
        number_of_peaks = len(peak_indices)
        # dictionary to store sample sizes and their fold indices
        folds_indices_dictionary_peak = {}
        folds_indices_dictionary_all = {}
        # iterate through sample sizes used for causal discovery
        for sample_size in conf.discovery.peak_sample_sizes:
            # print(f'running for sample size {sample_size}')
            # check if ths sample size is to large to be run on this catchment
            # this is the case if it is not possible to get the min_folds_per_basin of mutually exclusive subsets of peaks
            folds_peak = []
            if conf.discovery.min_folds_per_basin * sample_size < number_of_peaks:
                # calculate number of folds actually used for the given sample size
                max_folds_possible = number_of_peaks // sample_size
                number_of_folds_peak = min(
                    max_folds_possible, conf.discovery.max_folds_per_basin
                )
                # generate folds from the peak data
                folds_peak = split_list_into_subsets(
                    data=peak_indices,
                    num_subsets=number_of_folds_peak,
                    subset_size=sample_size,
                    random_seed=conf.discovery.sample_selection_seed,
                )

            # generate folds from random samples from the data (must not be peak)
            all_indices = basin.index.tolist()
            folds_all = split_list_into_subsets(
                data=all_indices,
                num_subsets=conf.discovery.max_folds_per_basin,
                subset_size=sample_size,
                random_seed=conf.discovery.sample_selection_seed,
            )

            # print(f'for sample size {sample_size} we have {len(folds_peak)} peak folds and {len(folds_all)} folds with all data')\
            # store fold indices in single dictionaries for submission to slurm cluster
            folds_indices_dictionary_peak[sample_size] = folds_peak
            folds_indices_dictionary_all[sample_size] = folds_all

        # create storage directory if it does not already exist
        # Create the directory to store the results if it does not exist
        os.makedirs(os.path.join(conf.io.models, "discovery_resampled"), exist_ok=True)
        # run for PCMCI with ParCorr on all folds for all sample sizes
        # set up submitit launcher
        executor = submitit.AutoExecutor(folder=conf.io.work)
        executor.update_parameters(
            timeout_min=conf.slurm.runtime_ParCorr,
            slurm_mem_per_cpu=conf.slurm.memory_ParCorr,
            cpus_per_task=1,
        )

        # run for peaks with ParCorr
        # send the job to be executed on the slurm cluster
        job = executor.submit(
            run_pcmciplus_full_basin,
            basin,  # data
            ParCorr,  # cond_ind_test
            "ParCorr",  # cond_ind_test_name
            conf.discovery.pc_alpha,  # pc_alpha
            folds_indices_dictionary_peak,  # fold_indices_dictionary
            link_assumptions,  # link_assumptions
            conf.discovery.tau_max,  # tau_max
            id,  # basin_id
            True,  # peaks_used
            os.path.join(conf.io.models, "discovery_resampled"),  # storage_path
            "Peak",  # storage_prefix
            conf.discovery.pcmci_verbosity,  # verbosity
        )
        print(f"Job {job.job_id} is ParCorr Peak run for basin {id}")

        # run for random samples with ParCorr
        # send the job to be executed on the slurm cluster
        job = executor.submit(
            run_pcmciplus_full_basin,
            basin,  # data
            ParCorr,  # cond_ind_test
            "ParCorr",  # cond_ind_test_name
            conf.discovery.pc_alpha,  # pc_alpha
            folds_indices_dictionary_all,  # fold_indices_dictionary
            link_assumptions,  # link_assumptions
            conf.discovery.tau_max,  # tau_max
            id,  # basin_id
            False,  # peaks_used
            os.path.join(conf.io.models, "discovery_resampled"),  # storage_path
            "All",  # storage_prefix
            conf.discovery.pcmci_verbosity,  # verbosity
        )
        print(f"Job {job.job_id} is ParCorr All run for basin {id}")

        # update slur job parameters for GPDC - use gpu and different time and memory requirements
        executor.update_parameters(
            timeout_min=conf.slurm.runtime_GPDC,
            slurm_mem_per_cpu=conf.slurm.memory_GPDC,
            cpus_per_task=1,
            gpus_per_node=1,
        )

        # run for peaks with GPDC
        # send the job to be executed on the slurm cluster
        job = executor.submit(
            run_pcmciplus_full_basin,
            basin,  # data
            GPDCtorch,  # cond_ind_test
            "GPDC",  # cond_ind_test_name
            conf.discovery.pc_alpha,  # pc_alpha
            folds_indices_dictionary_peak,  # fold_indices_dictionary
            link_assumptions,  # link_assumptions
            conf.discovery.tau_max,  # tau_max
            id,  # basin_id
            True,  # peaks_used
            os.path.join(conf.io.models, "discovery_resampled"),  # storage_path
            "Peak",  # storage_prefix
            conf.discovery.pcmci_verbosity,  # verbosity
        )
        print(f"Job {job.job_id} is GPDC Peak run for basin {id}")

        # run for random samples with GPDC
        # send the job to be executed on the slurm cluster
        job = executor.submit(
            run_pcmciplus_full_basin,
            basin,  # data
            GPDCtorch,  # cond_ind_test
            "GPDC",  # cond_ind_test_name
            conf.discovery.pc_alpha,  # pc_alpha
            folds_indices_dictionary_all,  # fold_indices_dictionary
            link_assumptions,  # link_assumptions
            conf.discovery.tau_max,  # tau_max
            id,  # basin_id
            False,  # peaks_used
            os.path.join(conf.io.models, "discovery_resampled"),  # storage_path
            "All",  # storage_prefix
            conf.discovery.pcmci_verbosity,  # verbosity
        )
        print(f"Job {job.job_id} is GPDC All run for basin {id}")

        # run_pcmciplus_full_basin(
        #     data = basin,
        #     cond_ind_test=ParCorr,
        #     cond_ind_test_name='ParCorr',
        #     pc_alpha=conf.discovery.pc_alpha,
        #     folds_indices_dictionary=folds_indices_dictionary_peak,
        #     link_assumptions=link_assumptions,
        #     tau_max=conf.discovery.tau_max,
        #     basin_id=id,
        #     peaks_used=True,
        #     storage_path=os.path.join(conf.io.models, 'discovery'),
        #     storage_prefix='Peak',
        #     verbosity=conf.discovery.pcmci_verbosity
        # )

    return


if __name__ == "__main__":
    main()
