import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import time

from linear_operator.utils.errors import NotPSDError, NanError

from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp

from discovery_result_dataclass import discovery_result


def mask_builder(data: pd.DataFrame, indices: list):
    """Creates a mask to only do causal discovery on the peaks

    Args:
        data (pd.DataFrame): Full input data
        indices (list): indices of the runoff values causal discovery

    Returns:
        numpy.array: boolean array masking out everything but the peaks
    """
    # mask must be same size as the data
    mask = np.full(data.shape, True)
    var_names = {"pre":0, "tavg7":1, "snow":2, "SM":3, "Q":4}
    mask_lags = {"pre":[0], "tavg7":[0], "snow":[7], "SM":[7], "Q":[0]}
    for var_name,var_index in var_names.items():
        lags = mask_lags[var_name]
        for lag in lags:
            for index in indices:
                mask[index-lag, var_index] = False
    return mask


def get_link_assumptions() -> dict:
    """Creates the link assumption dictionary used in this study

    Returns:
        dict: link assumption dictioinary formated {effect: {(cause,lag):direction_arrow}}
    """
    var_names = ["pre", "tavg7", "snow", "SM", "Q"]
    pre, tavg, snow, sm, q = 0, 1, 2, 3, 4
    link_assumptions = {effect: {} for effect in range(0, len(var_names))}
    link_assumptions[q] = {
        (q, -1): "-?>",
        (tavg, 0): "-?>",
        (sm, -7): "-?>",
        (snow, -7): "-?>",
        (pre, 0): "-?>",
        (pre, -1): "-?>",
        (pre, -2): "-?>",
        (pre, -3): "-?>",
        (pre, -4): "-?>",
        (pre, -5): "-?>",
        (pre, -6): "-?>",
    }
    link_assumptions[sm] = {
        (sm, -1): "-?>",
        (tavg, 0): "o?o",
        (pre, 0): "-?>",
        (snow, 0): "o?o",
    }
    link_assumptions[tavg] = {
        (tavg, -1): "-?>",
        (sm, 0): "o?o",
        (pre, 0): "o?o",
        (snow, 0): "o?o",
    }
    link_assumptions[snow] = {
        (snow, -1): "-?>",
        (sm, 0): "o?o",
        (tavg, 0): "o?o",
        (pre, 0): "o?o",
    }
    link_assumptions[pre] = {(pre, -1): "-?>", (tavg, 0): "o?o", (snow, 0): "o?o"}
    return link_assumptions


def split_list_into_subsets(
    data: list, num_subsets: int, subset_size: int, random_seed: int = 42
) -> list:
    """splits list into unique random subsets

    Args:
        data (list): input dataset
        num_subsets (int): number of subsets to be created
        subset_size (int): size of each subset
        random_seed (int, optional): random seed to be used for reproducibility. using system default if not set.

    Raises:
        ValueError: not enough data to create desired subsets (num_subsets * subset_size > len(data))

    Returns:
        list: random unique subsets as a list of lists
    """
    # ensure the list has enough elements
    if len(data) < num_subsets * subset_size:
        raise ValueError(
            "The list does not have enough elements to create the specified number of subsets."
        )
    # set random seed
    random.seed(random_seed)
    # select unique elements for the subsets
    selected_elements = random.sample(data, num_subsets * subset_size)
    subsets = [
        selected_elements[i : i + subset_size]
        for i in range(0, len(selected_elements), subset_size)
    ]
    return subsets


def run_pcmciplus(
    data: pd.DataFrame,
    cond_ind_test,
    pc_alpha: float,
    fold_indices: list,
    link_assumptions: dict,
    tau_max: int,
    verbosity: int = 0,
) -> dict:
    """Runs PCMCI_plus for a single dataset

    Args:
        data (pd.DataFrame): training data
        cond_ind_test (_type_): conditional independence test. In this study ParCorr or GPDC
        pc_alpha (float): PC hyperparameter
        fold_indices (list): indices of the runoff variable to be used in training
        link_assumptions (dict): link assumptions from domain knowledge
        tau_max (int): maximal time-lag used
        verbosity (int, optional): verbosity of pcmci. Defaults to 0.

    Returns:
        dict: pcmci/tigramite result dictionary with graph and p_value matrix
    """
    # create mask in tigramite format to perform causal discovery on the selected indices only
    mask = mask_builder(
        data=data, indices=fold_indices
    )
    # build tigramite dataframe
    dataframe = pp.DataFrame(
        data.to_numpy(), var_names=data.columns.to_list(), mask=mask
    )
    # create pcmci object with mask type y to have selected runoff indices beeing effect only
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test(significance="analytic", mask_type="y"),
        verbosity=verbosity,
    )
    results = pcmci.run_pcmciplus(
        tau_max=tau_max, pc_alpha=pc_alpha, link_assumptions=link_assumptions
    )
    return results


def run_pcmciplus_full_basin(
    data: pd.DataFrame,
    cond_ind_test,
    cond_ind_test_name: str,
    pc_alpha: float,
    folds_indices_dictionary: dict,
    link_assumptions: dict,
    tau_max: int,
    basin_id: int,
    peaks_used: bool,
    storage_path: str,
    storage_prefix: str,
    verbosity: int = 0,
):
    for sample_size, folds in folds_indices_dictionary.items():
        print(f"running for sample size {sample_size}")
        for fold_id, fold in tqdm(enumerate(folds)):
            try:
                # track time for causal discovery
                start_time = time.time()
                # do causal discovery for a single fold
                pcmci_result = run_pcmciplus(
                    data=data,
                    cond_ind_test=cond_ind_test,
                    pc_alpha=pc_alpha,
                    fold_indices=fold,
                    link_assumptions=link_assumptions,
                    tau_max=tau_max,
                    verbosity=verbosity,
                )

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"fold{fold_id:2d} took {elapsed_time:.2f}s")
                # put the results and all the metadata into a dataclass
                result = discovery_result(
                    basin_id=basin_id,
                    sample_size=sample_size,
                    fold_id=fold_id,
                    cond_ind_test=cond_ind_test_name,
                    indices_used=fold,
                    peaks_used=peaks_used,
                    discovery_result=pcmci_result,
                    discovery_runtime=elapsed_time,
                )

                # write results to disk
                result.save_to_disk(
                    storage_path=storage_path, storage_prefix=storage_prefix
                )
            # catch GP regression exception - continue with next fold
            except NotPSDError as e:
                print(
                    f"For basin {basin_id}, sample size {sample_size}, fold {fold_id}, GP regression was not possible for at least one step of the PC algorithm. ({str(e)})"
                )
                continue
            except NanError as e:
                print(
                    f"For basin {basin_id}, sample size {sample_size}, fold {fold_id}, GP regression was not possible for at least one step of the PC algorithm. ({str(e)})"
                )
                continue
    print(f'all folds done - returning')
    sys.exit(0)
    return


def run_pcmciplus_obs(
    data: pd.DataFrame,
    cond_ind_test,
    cond_ind_test_name: str,
    pc_alpha: float,
    indices: list,
    link_assumptions: dict,
    tau_max: int,
    basin_id: int,
    peaks_used: bool,
    storage_path: str,
    storage_prefix: str,
    verbosity: int = 0,
):
    # track time for causal discovery
    start_time = time.time()
    # do causal discovery for a single fold
    pcmci_result = run_pcmciplus(
        data=data,
        cond_ind_test=cond_ind_test,
        pc_alpha=pc_alpha,
        fold_indices=indices,
        link_assumptions=link_assumptions,
        tau_max=tau_max,
        verbosity=verbosity,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"discovery took {elapsed_time}")
    # put the results and all the metadata into a dataclass
    result = discovery_result(
        basin_id=basin_id,
        sample_size=len(indices),
        fold_id=0,
        cond_ind_test=cond_ind_test_name,
        indices_used=indices,
        peaks_used=peaks_used,
        discovery_result=pcmci_result,
        discovery_runtime=elapsed_time,
    )

    # write results to disk
    result.save_to_disk(
        storage_path=storage_path, storage_prefix=storage_prefix
    )
    sys.exit(0)
    return

if __name__ == '__main__':
    pass