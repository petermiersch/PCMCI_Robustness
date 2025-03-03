from dataclasses import dataclass, field
import os
import pickle


@dataclass
class discovery_result:
    """Dataclass to efficiently store the and analyse the results of causal discovery with tigramite/pcmci
    """
    basin_id:int = field(default=None)
    sample_size:int = field(default=None)
    fold_id:int = field(default=None)
    cond_ind_test:str = field(default=None)
    indices_used:list = field(default=None)
    peaks_used:bool = field(default=None)
    discovery_result:dict = field(default = None, repr = False)
    discovery_runtime:float = field(default = None, repr=False)

    def save_to_disk(self, storage_path: str, storage_prefix:str ):
        """Save the instance to a file."""
        filename = os.path.join(storage_path,f'{storage_prefix}_{self.cond_ind_test}_basin-{self.basin_id}_size-{self.sample_size:05d}_fold-{self.fold_id:03d}.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_disk(cls, file_path: str):
        """Create an instance from a file."""
        with open(file_path, 'rb') as file:
            return pickle.load(file)