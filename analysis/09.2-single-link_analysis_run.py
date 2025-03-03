import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.pcmci import PCMCI
import submitit

from preprocessing import moving_average
from causal_discovery_tools import get_link_assumptions, split_list_into_subsets, mask_builder


from itertools import combinations
from icecream import ic
import pandas as pd
from tqdm import tqdm
import os
import sys
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import time

import signal

# def cleanup(signum, frame): # Clean up resources here sys.exit(0)
#     sys.exit(0)

def run_single_cond_ind_test(cond_ind_test, X, Y, Z, alpha):
    test_statistic, p_value, dependent = cond_ind_test.run_test(
        X=X, 
        Y=Y,
        Z=Z,
        alpha_or_thres = alpha
        )
    # print(f"P-value for conditional independence: {p_value}")
    return test_statistic, p_value, dependent

def get_statistics_unconditional_independence(cond_ind_test, X, Ys, alpha):
    results = []  # List to store (Y, test_statistic, p_value) tuples
    for Y in Ys:
        # ic(Y)
        test_statistic, p_value, dependent = cond_ind_test.run_test(X= X, Y = [Y], Z = None,  alpha_or_thres = alpha)
        results.append((Y, abs(test_statistic), p_value, dependent))
    # Sort the results by test_statistic in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results

def all_combinations(unique_objects):
    result = []
    for r in range(1, len(unique_objects) + 1):  # r is the length of the combination
        result.extend(list(combinations(unique_objects, r)))
    return [list(comb) for comb in result]


def run_all_combinations_cond_ind_tests(cond_ind_test, X, Y, all_parents, alpha):
    Zs = all_combinations(all_parents)
    results = []
    for Z in Zs:
        test_statistic, p_value, dependent = run_single_cond_ind_test(cond_ind_test=cond_ind_test, X= X, Y = Y, Z = Z, alpha=alpha)
        results.append((Z, test_statistic, p_value, dependent))
    return(results)

def run_increasing_order_cond_ind_tests(cond_ind_test, X,Y, all_parents, alpha):
    # get the parents of X unconditional independence test statistic
    statistical_test_ordered_results = get_statistics_unconditional_independence(cond_ind_test=cond_ind_test, X = X, Ys = all_parents, alpha=alpha)
    # convert to list
    # remove Y from the list as it should not be used in the conditioning set
    all_parents_ordered = [l[0] for l in statistical_test_ordered_results if l[0] != Y[0]]
    # iterate through increasing size of conditioning set
    results_df = pd.DataFrame(columns=['cond_size', 'cond_set', 'test_statistic', 'p_value', 'dependent'])
    for cond_size in range(0,len(all_parents_ordered)):
        cond_set = all_parents_ordered[0:cond_size]
        test_statistic, p_value, dependent = run_single_cond_ind_test(cond_ind_test=cond_ind_test, X= X, Y = Y, Z = cond_set, alpha=alpha)
        row = {
        'cond_size': cond_size,
        'cond_set': cond_set,
        'test_statistic': test_statistic,
        'p_value': p_value,
        'dependent': int(dependent)
        }
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    return results_df


def run_increasing_order_cond_ind_tests_folds(cond_ind_test, X,Y, alpha, data, peaks, max_folds, subset_size, output_file = None):
    # calculate the maximum number of folds possible given the number of peaks and the fold size
    start_time = time.perf_counter() 
    max_folds_possible = len(peaks) // subset_size
    folds_used = min(max_folds,max_folds_possible)
    # create the folds
    folds_peak = split_list_into_subsets(
                    data=peaks,
                    num_subsets=folds_used,
                    subset_size=1000,
                    random_seed=1337,
                )
    basin_results = pd.DataFrame()
    # iterate through folds
    for fold_index in range(folds_used):
        # create the mask with the fold indices
        mask = mask_builder(data= data, indices=folds_peak[fold_index])

        # initilize dataframe and cond_ind_test
        dataframe = pp.DataFrame(data.to_numpy(), var_names=data.columns.to_list(), mask = mask)
        cond_ind_test.set_dataframe(dataframe)
        cond_ind_test.set_mask_type('y')
        # get all parents of q
        parents_of_q = list(get_link_assumptions()[4].keys())

        # get results for single fold
        results_fold = run_increasing_order_cond_ind_tests(cond_ind_test=cond_ind_test,
                                            X = X,
                                            Y = Y,
                                            all_parents=parents_of_q,
                                            alpha=alpha,
                                            )
        results_fold['fold'] = fold_index
        # concate the results into a single dataframe
        basin_results = pd.concat([basin_results, results_fold], ignore_index=True)
    # write out the results to csv
    if output_file is not None:
        basin_results.to_csv(output_file, index=False)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    # signal.signal(signal.SIGTERM, cleanup)
    return 0

ids = [
    6337610,
    6502171,
    6335046,
    6338200,
    6338160,
    6123710,
    6503851,
    6229100,
    6335351,
    6123350,
    6139682,
    6503280,
    6503500,
    6731600,
    6338120,
    6934571,
    6338161,
    6503855,
    6335160,
    6139790,
    6502151,
    6233350,
    6233100,
    6321100,
    6342521,
    6604220,
    6243400,
    6338150,
    6503281,
    6337504,
    6335045,
    6503201,
    6335081,
    6119200,
    6854590,
    6337050,
    6503180,
    6123160,
    6503351,
    6335360,
    6503301,
    6855409,
    6136200,
    6503300,
    6233520,
  ]

def main() -> None:
    
    # read the data
    data_base_path = '/data/compoundx/causal_flood/stability_testing/data/'
    data_path = os.path.join(data_base_path,'resampled')
    var_names = ["pre", "tavg7", "snow", "SM", "Q"]
    pre, tavg, snow, sm, q = 0, 1, 2, 3, 4


    for id in tqdm(ids):
        peak_indices = []
        with open(os.path.join(data_path, f'{id}_peak_indices_USWRC.txt'), 'r') as filehandle:
            peak_indices = [current_place.rstrip() for current_place in filehandle.readlines()]
        peak_indices = list(map(int, peak_indices))
        data = pd.read_csv(os.path.join(data_path, f'{id}.csv'))
        data['time'] = data['time'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        data["tavg7"] = moving_average(data["tavg"], 7)
        # data = data.set_index('time')
        data = data[["pre", "tavg7", "snow", "SM", "Q"]]
        ic(id)

        X = [(q,0)]
        Y = [(pre,-4)]

        # single_basin_results = run_increasing_order_cond_ind_tests_folds(
        #     cond_ind_test = ParCorr(),
        #     X = X,
        #     Y = Y,
        #     alpha = 0.05,
        #     data = all_data[context][id],
        #     peaks = all_peaks[context][id],
        #     max_folds = 10,
        #     subset_size = 1000,
        #     output_file = os.path.join('/data/compoundx/causal_flood/stability_testing/single_edge_analysis/ParCorr/', f'{id}.csv'))



        executor = submitit.AutoExecutor(folder='/work/miersch/submitit')
        executor.update_parameters(
            timeout_min=10,
            slurm_mem_per_cpu="8G",
            cpus_per_task=1,
            gpus_per_node=1,
        )
        output_file = os.path.join('/data/compoundx/causal_flood/stability_testing/single_edge_analysis/GPDC_pre4/', f'{id}.csv')
        if not os.path.exists(output_file):
        # run for peaks with ParCorr
        # send the job to be executed on the slurm cluster
            job = executor.submit(
                run_increasing_order_cond_ind_tests_folds,
                cond_ind_test = GPDCtorch(),
                X = X,
                Y = Y,
                alpha = 0.05,
                data = data,
                peaks = peak_indices,
                max_folds = 10,
                subset_size = 1000,
                output_file = output_file
            )
            print(f"Job {job.job_id} is running for basin {id}")



if __name__ == "__main__":
    main()