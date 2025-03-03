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

from discovery_result_dataclass import discovery_result


def calculate_number_of_links_in_graph(graph: np.ndarray) -> int:
    """calculates the number of discovered links in a graph discovered by tigramite/pcmci

    Args:
        graph (np.ndarray): graph in the tigramite format

    Returns:
        int: number of links in that graph
    """
    # remove above diagonal links to avoid double-counting
    graph_adjusted = set_above_diagonal_to_empty(graph)
    number_of_links = np.sum(graph != "")
    return number_of_links


def set_above_diagonal_to_empty(graph: np.ndarray) -> np.ndarray:
    """for a tigramite result graphs, sets all elements above the diagonal line for tau = 0 to empty string to avoid double counting when comparing to other graphs

    Args:
        graph (np.ndarray): tigramit result graph

    Returns:
        np.ndarray: tigramite result graphs with empty strings above the diagonal line
    """
    fixed_graph = graph
    t = 0
    graph_slice = graph[:, :, t]
    rows, columns = graph_slice.shape
    for row in range(rows):
        for column in range(row + 1, columns):
            graph_slice[row, column] = ""
    fixed_graph[:, :, t] = graph_slice
    return fixed_graph


def comparison_matrix_graph(graph1: np.ndarray, graph2: np.ndarray) -> np.ndarray:
    """comparison matrix between two tigramite result graphs. Elements are False if link is the same, True if different.
    to calculate distance between graphs, elements above the diagonal should be regarded to not count links twice

    Args:
        graph1 (np.ndarray): tigramite result graph
        graph2 (np.ndarray): tigramite result graph

    Returns:
        np.ndarray: False if links are the same, True if not. Same shape as imput graph.
    """
    matrix = graph1 != graph2
    return matrix


def distance_between_graphs(
    graph1: np.ndarray, graph2: np.ndarray, normalize: bool = False
):
    """calculates the distance between two graphs by the number of links that are not the same

    Args:
        graph1 (np.ndarray): tigramite result graph
        graph2 (np.ndarray): tigramite result graph
        normalize (bool): if the distance should be normalized my the mean number of links in the two graphs

    Returns:
        float: distance between graphs as number of links not equal
    """
    # first set all elements above the diagonal line to empty string to avoid double counting
    graph1 = set_above_diagonal_to_empty(graph1)
    graph2 = set_above_diagonal_to_empty(graph2)
    # get comparison matrix
    matrix = comparison_matrix_graph(graph1, graph2)
    # distance are the number of links  that are not equal between the two graphs
    distance = np.sum(matrix)
    # normalize by the mean number of links of the two graphs
    if normalize:
        number_of_links1 = np.sum(graph1 != "")
        number_of_links2 = np.sum(graph2 != "")
        average_number_of_links = (number_of_links1 + number_of_links2) / 2
        distance = distance / average_number_of_links
    return distance


def graph_distance_distribution_between_model_list(
    models: list, normalize=False
) -> list:
    """calculates the (normalized) distance between graphs of all models

    Args:
        models (list):  list of tigramite result graphs
        normalize (bool, optional): if the distance should be normalized my the mean number of links in the two graphs. Defaults to False.

    Returns:
        list: list of distances between all graphs
    """
    distances = []
    graphs = [m.results["graph"] for m in models]
    for i, graph1 in enumerate(graphs):
        for graph2 in graphs[i + 1 :]:
            distance = distance_between_graphs(graph1, graph2, normalize=normalize)
            distances.append(distance)
    return distances


@hydra.main(version_base=None, config_path="conf", config_name="stability_analysis")
def main(conf: DictConfig) -> None:
    # load the discovery results from the previous steps into a single list
    all_files = os.listdir(os.path.join(conf.io.models, "discovery_resampled"))
    # Filter the list to include only .pkl files in case there are other files in the directory
    pkl_files = [file for file in all_files if file.endswith(".pkl")]
    discovery_results = []
    print(f"loading a total of {len(pkl_files)} causal discovery results")
    # read all files
    for file in tqdm(pkl_files):
        file_path = os.path.join(conf.io.models, "discovery_resampled", file)
        result = discovery_result.load_from_disk(file_path=file_path)
        discovery_results.append(result)
    # create list of all unique values in our discovered results. used for subselection of the list
    # basin_ids = list(set(model.basin_id for model in discovery_results))
    # sample_sizes = list(set(model.sample_size for model in discovery_results))
    # cond_ind_tests_used = list(set(model.cond_ind_test for model in discovery_results))
    # peaks_used = list(set(model.peaks_used for model in discovery_results))
    # print(basin_ids)
    # print(sample_sizes)
    # print(cond_ind_tests_used)
    # print(peaks_used)
    # store all comparison metrics for each model in a long list, basin_id, sample_size, fold_id, cond_ind_test_used, peaks_used, uniquely identify each discovery result
    dtypes = {
        "basin_id": "int32",
        "sample_size": "int32",
        "fold_id": "int32",
        "cond_ind_test": "str",
        "peaks_used": "bool",
        "discovery_runtime": "float",
        "number_of_links": "int32",
        "number_of_uncertain_links": "float",
        "fraction_of_uncertain_links": "float",
    }
    long_comparison_results = pd.DataFrame(
        {col: pd.Series(dtype=typ) for col, typ in dtypes.items()}
    )
    print(long_comparison_results.dtypes)
    # iterate through all discovery results (=models)
    print(
        f"iterating through all {len(discovery_results)} results to calculate metrics"
    )
    for model_index, model in tqdm(enumerate(discovery_results)):
        # calculate the number of links in graph
        number_of_links = calculate_number_of_links_in_graph(
            model.discovery_result["graph"]
        )
        # find models with same basin_id, sample_size, cond_ind_test, and peaks_used to compare the result to
        comparison_models = [
            m
            for m in discovery_results
            if m.basin_id == model.basin_id
            and m.sample_size == model.sample_size
            and m.cond_ind_test == model.cond_ind_test
            and m.peaks_used == model.peaks_used
        ]
        # calculate the number of links that are different to each model in the set of comparison models
        number_of_different_links_to_comparison_models = []
        for comparison_model in comparison_models:
            distance = distance_between_graphs(
                model.discovery_result["graph"],
                comparison_model.discovery_result["graph"],
                normalize=False,
            )
            number_of_different_links_to_comparison_models.append(distance)
        mean_number_of_different_links_to_comparison_models = np.mean(
            number_of_different_links_to_comparison_models
        )
        # write the results to the comparison list

        long_comparison_results.loc[model_index, "basin_id"] = int(model.basin_id)
        long_comparison_results.loc[model_index, "sample_size"] = int(model.sample_size)
        long_comparison_results.loc[model_index, "fold_id"] = int(model.fold_id)
        long_comparison_results.loc[model_index, "cond_ind_test"] = model.cond_ind_test
        long_comparison_results.loc[model_index, "peaks_used"] = bool(model.peaks_used)
        long_comparison_results.loc[model_index, "discovery_runtime"] = float(
            model.discovery_runtime
        )
        long_comparison_results.loc[model_index, "number_of_uncertain_links"] = float(
            mean_number_of_different_links_to_comparison_models
        )
        long_comparison_results.loc[model_index, "number_of_links"] = int(
            number_of_links
        )
    # calculate fraction of links by dividing by number of links for selected model
    long_comparison_results["fraction_of_uncertain_links"] = (
        long_comparison_results["number_of_uncertain_links"]
        / long_comparison_results["number_of_links"]
    )

    print(long_comparison_results)
    # define the data types of the comparison result list to write and read them to csv properly
    dtypes_df = pd.DataFrame(list(dtypes.items()), columns=["column", "dtype"])
    dtypes_df.to_csv(
        os.path.join(conf.io.models, "comparison_results_dtypes.csv"), index=False
    )

    # write the comparison results to disk
    long_comparison_results.to_csv(
        os.path.join(conf.io.models, "comparison_results.csv")
    )
    return


if __name__ == "__main__":
    main()
