# pcmci master class. preparing pcmci with data, mask and link assumptions, running it, and storing the output
from dataclasses import dataclass, field
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci_base import PCMCIbase
from tigramite.models import Prediction
import pandas as pd
import numpy as np
import sklearn
import pickle
from typing import Literal
from preprocessing import get_mask, get_index_from_name

@dataclass()
class pcmci_master:
    data: pd.DataFrame = field(repr=False)
    tau_max: int
    cond_ind_test: CondIndTest
    target: str = None
    cond_ind_test_significance: str = 'analytic'
    mask_type: str = None
    pcmci_version: Literal['pcmci','pcmciplus' ] = 'pcmci'
    percentile: int = None
    threshold: float = None
    pc_alpha: float = None
    extremes_declustering_range: int = 5
    mask: np.array = field(repr=False, default= None)
    dataframe: pp.DataFrame = field(init = False ,repr = False)
    link_assumptions: dict = field(repr = False, default = None)
    pcmci:PCMCI = field(init=False, repr=False)
    results:dict = field(init=False, repr = False)
    # dataframe = pp.DataFrame = None


    def __post_init__(self) -> None:
        # make mask according to threshold or percentile if mask is required (mask_type is not None)
        if (self.mask == None) & (self.mask_type != None):
            if ((self.threshold == None) & (self.percentile == None)) | (self.target == None):
                raise(ValueError('Mask could not be created. Provide target and threshold or percentile for mask creation or provide mask'))
            # make threshold from percentile
            if self.threshold == None:
                self.threshold = np.percentile(self.data[self.target],self.percentile)
            # make mask from threshold
            self.mask = get_mask(data = self.data,
                                target_name = self.target,
                                threshold = self.threshold,
                                extremes_type = 'high',
                                r = self.extremes_declustering_range)
        
        # make tigramite dataframe
        self.dataframe = pp.DataFrame(self.data.to_numpy(), 
                             datatime = np.arange(len(self.data)), 
                             var_names=self.data.columns,
                             mask = self.mask)
    def do_causal_discovery(self) -> None:
        """Performs causal discovery with given data, masks, and parameters
        """
        self.pcmci = PCMCI(dataframe=self.dataframe, 
                           cond_ind_test=self.cond_ind_test(significance=self.cond_ind_test_significance,
                           mask_type = self.mask_type),
                           verbosity=0)
        if self.pcmci_version == 'pcmci':
            self.results = self.pcmci.run_pcmci(tau_max=self.tau_max, pc_alpha=self.pc_alpha, link_assumptions=self.link_assumptions)
        elif self.pcmci_version == 'pcmciplus':
            self.results = self.pcmci.run_pcmciplus(tau_max=self.tau_max, pc_alpha=self.pc_alpha, link_assumptions=self.link_assumptions)
        else:
            raise ValueError(f'{self.pcmci_version} is not a supported version of pcmci')
        return
    def get_parents(self, include_lagzero_parents=True) -> dict:
        """Returns dict of parents estimated by pcmci

        Parameters
        ----------
        include_lagzero_parents : bool, optional
            If 0 lag parents should be included. Must be false for linear mediation, by default False

        Returns
        -------
        dict
            Dictionary of estimated parents used for linear mediation and prediction
        """
        estimated_parents = self.pcmci.return_parents_dict(self.results['graph'],
                                                           self.results['val_matrix'],
                                                           include_lagzero_parents=include_lagzero_parents)
        return estimated_parents
    def get_number_events(self) -> int:
        """Get number of unmasked events of the effect

        If there is no mask, this is equal to the length of the input data

        Returns
        -------
        int
            number of events
        """
        if type(self.mask) == type(None):
            return len(self.data)
        else:
            return sum(np.invert(self.mask[:,self.mask.shape[1]-1]))

    def plot_graph(self, **kwargs):
        """Creating a graph plot of the results

        Returns
        -------
        fig, ax
            graph plot figure and axis
        """
        return tp.plot_graph(val_matrix=self.results['val_matrix'],
                        graph=self.results['graph'],
                        var_names=self.data.columns,
                        link_colorbar_label='cross-MCI',
                        node_colorbar_label='auto-MCI',
                        **kwargs)

    def plot_time_series_graph(self, **kwargs):
        """Creating a time series graph plot of the results

        Returns
        -------
        fig, ax
            graph plot figure and axis
        """
        return tp.plot_time_series_graph(
            val_matrix=self.results['val_matrix'],
            graph=self.results['graph'],
            var_names=self.data.columns,
            arrow_linewidth=2,
            **kwargs)

    def get_link_dictionary(self) -> dict:
        """returns the dictionary of the discovered links
        warapper around the tigramite graph to dict function

        Returns
        -------
        dict
            link_dictionary {effect:{(cause, time_lag):'link type',...},...}
        """
        return PCMCIbase.graph_to_dict(self.results['graph'])

    def get_predictors(self, target:str) -> dict:
        """get the predictors as estimated by tigramite as dictionary used in prediction class

        Parameters
        ----------
        target : str
            variable name of the target

        Returns
        -------
        dict
            dictionary of the estimated predictors
        """
        target_int = get_index_from_name(self.dataframe, target)
        link_dictionary = self.get_link_dictionary()
        predictors = {target_int:[link for link in link_dictionary[target_int] if link_dictionary[target_int][link] == '-->']}
        return predictors

def predict(self, target, model, train_indices, test_indices) -> np.ndarray:
    target_id = get_index_from_name(target)
    predictors = self.get_predictors(target)
    pred = Prediction(dataframe=self.dataframe,
                prediction_model = model,
                data_transform=sklearn.preprocessing.StandardScaler(),
                train_indices= train_indices,
                test_indices= test_indices
                )
    pred.fit()
