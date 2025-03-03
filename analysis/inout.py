import pandas as pd
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

def get_filenames(directory:str):
    """Get names of all files in a directory
    
    Parameters
    -----
    directory: str
        input directory
    
    Returns
    -----
    file names: list(str)
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f))]
    return files

def get_ids(directory:str):
    """Get ids from names of basin averaged files in directory
    
    Parameters
    -----
    directory: str
        input directory
    
    Returns
    -----
    ids: list(int)
    """
    files = get_filenames(directory)
    ids = [int(i.split('.')[0]) for i in files]
    return ids

def read_single_basin(directory,id,filetype = '.csv', drop_unobserved = False):
    """Read single basin into pandas dataframe
    
    Parameters
    -----
    directory: str
        input directory
    id: int
        id of the basin (corresponds to filename)
    filetype: str
        filetype of the basin file
    drop_unobserved: bool
        if period without observed runoff should be dropped

    Returns
    -----
    basins: pandas.DataFrame
    """
    file = os.path.join(directory,str(id) + filetype)
    basin = pd.read_csv(file)

    if drop_unobserved:
        basin = basin.dropna(subset = ['Qobs'])
    return basin

def read_all_basins(directory,filetype = '.csv', drop_unobserved = False):
    """Read all basin data in directory into dictionary
    
    Parameters
    -----
    directory: str
        input directory
    drop_unobserved: bool
        if period without observed runoff should be dropped
    
    Returns
    -----
    basins: dict(pandas.DataFrame)
    """
    ids = get_ids(directory)
    basins = {}
    for i in tqdm(range(0,len(ids))):
        id = ids[i]
        basins[id] = read_single_basin(directory,id,filetype,drop_unobserved)
    return basins

def get_folder_names_as_int(directory:str) -> list:
    """Get all folder names as intergers if possible

    Args:
        directory (str): direcotry 

    Returns:
        list: list of all folder names as integers
    """
    try:
        return [int(name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    except ValueError:
        print("One or more folder names could not be converted to integers.")
        return None
if __name__ == '__main__':
    pass
