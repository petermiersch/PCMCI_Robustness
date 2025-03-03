import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pandas as pd
import submitit

from inout import read_single_basin
from preprocessing import moving_average, get_extremes_USWRC


def get_peaks_and_save(id: int, area: float, data: str):
    # read basin data
    basin = read_single_basin(directory=data, id=id)
    # get the peak indices for the full timeseries using the USWRC peak definition and the catchment area from the grdc database
    peak_indices = get_extremes_USWRC(basin["Q"], area)
    # export the peak indices to a text file in the same directory as the input data
    with open(os.path.join(data, f"{id}_peak_indices_USWRC.txt"), "w") as filehandle:
        filehandle.writelines(f"{peak}\n" for peak in peak_indices)


@hydra.main(version_base=None, config_path="conf", config_name="peak_detection")
def main(conf: DictConfig) -> None:
    # read the basin information from as reported in the grdc database
    grdc_info = pd.read_excel(conf.io.grdc_info)
    grdc_info = grdc_info.rename(columns={"grdc_no": "id"})
    grdc_info = grdc_info.set_index("id")
    # read basin timeseries and extract peaks from it
    for id in tqdm(conf.io.ids):
        # set up submitit launcher
        executor = submitit.AutoExecutor(folder=conf.io.work)
        executor.update_parameters(
            timeout_min=30, slurm_mem_per_cpu="4G", cpus_per_task=1
        )
        # get the peak indices for the full timeseries using the USWRC peak definition and the catchment area from the grdc database
        area = grdc_info["area"].loc[id] * 0.38610216  # area conversion to square miles
        # send the job to be executed on the slurm cluster for observed, simulated, and resampled data
        job = executor.submit(get_peaks_and_save, id, area, os.path.join(conf.io.data, 'observed'))
        print(job.job_id)
        job = executor.submit(get_peaks_and_save, id, area,  os.path.join(conf.io.data, 'simulated'))
        print(job.job_id)
        job = executor.submit(get_peaks_and_save, id, area,  os.path.join(conf.io.data, 'resampled'))
        print(job.job_id)
    return


if __name__ == "__main__":
    main()
