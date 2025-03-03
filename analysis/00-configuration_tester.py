import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import pandas as pd
import submitit


@hydra.main(version_base=None, config_path="conf", config_name="peak_detection")
def main(conf: DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))
    return


if __name__ == "__main__":
    main()
