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
from icecream import ic

from causal_discovery_tools import get_link_assumptions, split_list_into_subsets, mask_builder, run_pcmciplus, run_pcmciplus_full_basin, run_pcmciplus_obs

def get_peak_indices(file:str) -> list:
    """Gets the peak indices from a txt file and reads them into a list

    Args:
        file (str): path to the file containing the peak indices
    """
    with open(
            file
        ) as filehandle:
            peaks = [
                current_place.rstrip() for current_place in filehandle.readlines()
            ]
    return list(map(int, peaks))

def are_peaks_local_maxima(data:pd.Series, peaks:list[int]) -> bool:
    """Checks if the peaks (integer indices) are local maximum of the data 

    Args:
        data (pd.Series): Series of values
        peaks (list[int]): peak indices of series

    Returns:
        bool: True if all peaks are local maxima, otherwise False
    """
    for peak in peaks:
        # Check left neighbor
        if peak > 0 and data[peak] <= data[peak - 1]:
            return False
        # Check right neighbor
        if peak < len(data) - 1 and data[peak] <= data[peak + 1]:
            return False
    return True


def check_all_datasets_peaks(
    datasets: list[pd.Series],
    peaks_list: list[list[int]]
) -> bool:
    """Checks if all given peaks in each dataset are local maxima.
    
    Args:
        datasets (list[pd.Series]): List of Series datasets.
        peaks_list (list[list[int]]): List of peak index lists, one per dataset.
        
    Returns:
        bool: True if all peaks in each dataset are local maxima, otherwise raises ValueError.
    """
    if len(datasets) != len(peaks_list):
        raise ValueError("The number of datasets must match the number of peaks lists.")
    
    for i, (data, peaks) in enumerate(zip(datasets, peaks_list)):
        if not are_peaks_local_maxima(data, peaks):
            raise ValueError(f"Dataset {i} has peaks that are not local maxima.")
    
    return True


def prepare_single_basin(basin:pd.DataFrame) -> pd.DataFrame:
    """Prepares the dataframe for analysis with PCMCI
    Only selects the variables ["pre", "tavg7", "snow", "SM", "Q"] where tavgz is the 7 day moving average of tavg.
    Also fills na with 0 as this is a requirement for tigramite. this is important for the observational dataset (it is the only one that contains na and only for the runoff Q).
    Peaks are only calculated on the non-na values, the na values that are replaced by 0s will not be used in analysis

    Args:
        basin (pd.DataFrame): raw datafram of basin data

    Returns:
        pd.DataFrame: dataframe only with relevant variables
    """
    basin['tavg7'] =  moving_average(basin["tavg"], 7)
    basin = basin[["pre", "tavg7", "snow", "SM", "Q"]]
    basin = basin.fillna(0)
    return basin

@hydra.main(version_base=None, config_path="conf", config_name="causal_discovery")
def main(conf: DictConfig) -> None:
    # for debugging of slurm jobs where a real debugger can not be used. control if program is verbose and writes lots of output when generating jobs
    verbose = False
    # Data directories ofr observed, simulated, and resampled dataset
    observed_data_directory = os.path.join(conf.io.data, 'observed')
    simulated_data_directory = os.path.join(conf.io.data, 'simulated')
    resampled_data_directory = os.path.join(conf.io.data, 'resampled')

    # Make model output directories for all 4 experiments
    observed_output_directory = os.path.join(conf.io.models, "discovery_observed")
    os.makedirs(observed_output_directory, exist_ok=True)
    simulated_output_directory = os.path.join(conf.io.models, "discovery_simulated")
    os.makedirs(simulated_output_directory, exist_ok=True)
    resampled_as_observed_output_directory = os.path.join(conf.io.models, "discovery_resampled_as_observed")
    os.makedirs(resampled_as_observed_output_directory, exist_ok=True)
    resampled_as_simulated_output_directory = os.path.join(conf.io.models, "discovery_resampled_as_simulated")
    os.makedirs(resampled_as_simulated_output_directory, exist_ok=True)

    # model output direcories of
    for id in conf.io.ids:
        ic(id)
        # get peak indices for all datasets
        peaks_observed = get_peak_indices(os.path.join(observed_data_directory, f"{id}_peak_indices_USWRC.txt"))
        peaks_simulated = get_peak_indices(os.path.join(simulated_data_directory, f"{id}_peak_indices_USWRC.txt"))
        peaks_resampled = get_peak_indices(os.path.join(resampled_data_directory, f"{id}_peak_indices_USWRC.txt"))
        number_of_peaks_observed = len(peaks_observed)
        number_of_peaks_simulated = len(peaks_simulated)
        number_of_peaks_resampled = len(peaks_resampled)
        if verbose:
            print(f'''basin {id} has peaks:
                    \t observed: \t {number_of_peaks_observed} 
                    \t simulated: \t {number_of_peaks_simulated} 
                    \t resampled: \t {number_of_peaks_resampled} ''')

        # get data for all datasets for basin
        basin_observed = prepare_single_basin(read_single_basin(directory=observed_data_directory, id=id))
        basin_simulated = prepare_single_basin(read_single_basin(directory=simulated_data_directory, id=id))
        basin_resampled = prepare_single_basin(read_single_basin(directory=resampled_data_directory, id=id))

        # check if peaks are local maxima. test if either peak detection went wrong or peaks where not loaded for the corresponding data
        try:
            check_all_datasets_peaks(
                [basin_observed['Q'], basin_simulated['Q'], basin_resampled['Q']],
                [peaks_observed, peaks_simulated, peaks_resampled]
            )
            if verbose:
                print(f'peaks for {id} are local maxima in all datasets')
        except ValueError as e:
            ic(e)

        # set up executor for slurm job (runnin on high performace cluster)
        # this part of the code only runs on a slurm cluster. the functions called (e.e. run_pcmciplus_obs) can also be run without slurm
        executor = submitit.AutoExecutor(folder=conf.io.work)
        executor.update_parameters(
            timeout_min=60,
            slurm_mem_per_cpu=conf.slurm.memory_GPDC,
            cpus_per_task=1,
            gpus_per_node=1
        )

        # run causal discovery for the observed dataset
        job = executor.submit(
            run_pcmciplus_obs,
            basin_observed,  # data
            GPDCtorch,  # cond_ind_test
            "GPDC",  # cond_ind_test_name
            conf.discovery.pc_alpha,  # pc_alpha
            peaks_observed,  # indices accounted for the shift in the observational data indices du to na removal
            get_link_assumptions(),  # link_assumptions
            conf.discovery.tau_max,  # tau_max
            id,  # basin_id
            True,  # peaks_used
            observed_output_directory,  # storage_path
            "Peak",  # storage_prefix
            conf.discovery.pcmci_verbosity,  # verbosity
        )
        print(f"Job {job.job_id} is causal discovery with PCMCI+ using GPDC on peaks for observed data in basin {id}.")

        # run causal discovery for the simulate dataset
        job = executor.submit(
            run_pcmciplus_obs,
            basin_simulated,  # data
            GPDCtorch,  # cond_ind_test
            "GPDC",  # cond_ind_test_name
            conf.discovery.pc_alpha,  # pc_alpha
            peaks_simulated,  # indices accounted for the shift in the observational data indices du to na removal
            get_link_assumptions(),  # link_assumptions
            conf.discovery.tau_max,  # tau_max
            id,  # basin_id
            True,  # peaks_used
            simulated_output_directory,  # storage_path
            "Peak",  # storage_prefix
            conf.discovery.pcmci_verbosity,  # verbosity
        )
        print(f"Job {job.job_id} is causal discovery with PCMCI+ using GPDC on peaks for simulated data in basin {id}.")


        # create 10 subsets of the resampled dataset that have the same size as the observed and simulated dataset for comparison of the resulting graphs
        # to use the same function to run the jobs as for the sample size sensitivity analysis, convert into dictionary with only one entry
        folds_peak_resampled_as_observed = {number_of_peaks_observed: split_list_into_subsets(
                data=peaks_resampled,
                num_subsets=10,
                subset_size=number_of_peaks_observed,
                random_seed=conf.discovery.sample_selection_seed,
            )}
        
        folds_peak_resampled_as_simulated = {number_of_peaks_simulated:split_list_into_subsets(
                data=peaks_resampled,
                num_subsets=10,
                subset_size=number_of_peaks_simulated,
                random_seed=conf.discovery.sample_selection_seed,
            )}
            
        # update executor to allow more time for the slurm jobs
        executor.update_parameters(
            timeout_min=150,
            slurm_mem_per_cpu=conf.slurm.memory_GPDC,
            cpus_per_task=1,
            gpus_per_node=1,
        )

        # run job for the resampled dataset with the same amount of data as the observed dataset
        job = executor.submit(
            run_pcmciplus_full_basin,
            basin_resampled,  # data
            GPDCtorch,  # cond_ind_test
            "GPDC",  # cond_ind_test_name
            conf.discovery.pc_alpha,  # pc_alpha
            folds_peak_resampled_as_observed,  # fold_indices_dictionary
            get_link_assumptions(),  # link_assumptions
            conf.discovery.tau_max,  # tau_max
            id,  # basin_id
            True,  # peaks_used
            resampled_as_observed_output_directory,  # storage_path
            "Peak",  # storage_prefix
            conf.discovery.pcmci_verbosity,  # verbosity
        )
        print(f"Job {job.job_id} is causal discovery with PCMCI+ using GPDC on peaks for resampled data with the same length as the simulated dataset in basin {id}.")

        # run job for the resampled dataset with the same amount of data as the simulated dataset
        job = executor.submit(
            run_pcmciplus_full_basin,
            basin_resampled,  # data
            GPDCtorch,  # cond_ind_test
            "GPDC",  # cond_ind_test_name
            conf.discovery.pc_alpha,  # pc_alpha
            folds_peak_resampled_as_simulated,  # fold_indices_dictionary
            get_link_assumptions(),  # link_assumptions
            conf.discovery.tau_max,  # tau_max
            id,  # basin_id
            True,  # peaks_used
            resampled_as_simulated_output_directory,  # storage_path
            "Peak",  # storage_prefix
            conf.discovery.pcmci_verbosity,  # verbosity
        )
        print(f"Job {job.job_id} is causal discovery with PCMCI+ using GPDC on peaks for resampled data with the same length as the simulated dataset in basin {id}.")

    return

if __name__ == "__main__":
    main()
    
