# preprocessing for pcmci
import logging
import warnings

from typing import Any, Generator, Literal, Union

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
import numpy.ma as ma
import copy

def standardize_dataframe(dataframe:pp.DataFrame) -> pp.DataFrame:
    """ 
    z-score normalization of tigramite dataframe over unmasked data

    Parameters
    -----
    dataframe: tigramite.data_processing.DataFrame
        unstandardized, potentially masked dataframe

    Retruns
    -----
    standardized_dataframe: tigramite.data_processing.DataFrame
    """
    data = ma.masked_array(data = dataframe.values[0], mask = dataframe.mask[0])
    mean = data.mean(axis = 0)
    std = data.std(axis = 0)
    standardized_data = (data - mean)/std
    standardized_dataframe = copy.deepcopy(dataframe)
    standardized_dataframe.values[0] = standardized_data.data
    return standardized_dataframe


# This is an adaption of pyextremes get_extremes function for series without date index
# both pyextremes and this software ware based on it can only be published using an MIT license
# this is necessary as pyextremes relies on datetime64 and pandas timedelta, which is only defined between ~1600 - 2500 (not of 1000 to 2000 being the range of the resampled data)
logger = logging.getLogger(__name__)
def _generate_clusters(
    exceedances: pd.Series,
    r: Union[pd.Timedelta, Any],
):
    # There can be no clusters if there are no exceedances
    if len(exceedances) == 0:
        return
    # There can be only one cluster if there is only one exceedance
    if len(exceedances) == 1:
        yield exceedances
        return

    # Locate clusters separated by gaps not smaller than `r`
    gap_indices = np.argwhere(
        (exceedances.index[1:] - exceedances.index[:-1]) > r
    ).flatten()
    if len(gap_indices) == 0:
        # All exceedances fall within the same cluster
        yield exceedances
    else:
        for i, gap_index in enumerate(gap_indices):
            if i == 0:
                # First cluster contains all values left from the gap
                yield exceedances.iloc[: gap_index + 1]
            else:
                # Other clusters contain values between previous and current gaps
                yield exceedances.iloc[gap_indices[i - 1] + 1 : gap_index + 1]

        # Last cluster contains all values right from the last gap
        yield exceedances.iloc[gap_indices[-1] + 1 :]

def get_extremes_peaks_over_threshold(
    ts: pd.Series,
    extremes_type: Literal["high", "low"],
    threshold: float,
    r: int,
) -> pd.Series:
    """
    Get extreme events from time series using the Peaks Over Threshold method.

    Parameters
    ----------
    ts : pandas.Series
        Time series of the signal.
    extremes_type : str
        high - get extreme high values (above threshold)
        low - get extreme low values (below threshold)
    threshold : float
        Threshold used to find exceedances.
    r : int duration of window used to decluster the exceedances in timesteps

    Returns
    -------
    extremes : pandas.Series
        Time series of extreme events.

    """
    logger.debug(
        "collecting peaks-over-threshold extreme events using "
        "extremes_type=%s, threshold=%s, r=%s",
        extremes_type,
        threshold,
        r,
    )

    if extremes_type not in ["high", "low"]:
        raise ValueError(
            f"invalid value in '{extremes_type}' for the 'extremes_type' argument"
        )

    # Get exceedances
    if extremes_type == "high":
        exceedances = ts.loc[ts.values > threshold]
    else:
        exceedances = ts.loc[ts.values < threshold]
    logger.debug("found %d exceedances", len(exceedances))

    if len(exceedances) == 0:
        warnings.warn(
            f"Threshold value '{threshold}' is too {extremes_type} "
            f"and results in zero extreme values"
        )

    # Locate clusters separated by gaps not smaller than `r`
    # and select min or max (depending on `extremes_type`) within each cluster
    extreme_indices, extreme_values = [], []
    for cluster in _generate_clusters(exceedances=exceedances, r=r):
        extreme_indices.append(
            cluster.idxmax() if extremes_type == "high" else cluster.idxmin()
        )
        extreme_values.append(cluster.loc[extreme_indices[-1]])

    logger.debug(
        "successfully collected %d extreme events",
        len(extreme_values),
    )
    return pd.Series(
        data=extreme_values,
        index=pd.Index(data=extreme_indices, name=ts.index.name or "date-time"),
        dtype=np.float64,
        name=ts.name or "extreme values",
    )

def get_extremes_USWRC(discharge:list, basin_area:float) -> list:
    """Extract peaks from  discharge timeseries
    follows the procedure recommended by the guidelines of the US Water Resources Council
    Peaks have to satisfy to cirteria:
    - minimum distance criterion: peaks must be 5 + log(A) apart where A ist the basin are in square miles
    - minimum discharge cirterion: discharge must fall below 75% of the lower peak between every consequent pair of peaks, if not the smaller peak is removed

    Parameters
    ----------
    discharge : list
        discharge timeseries
    basin_area : float
        area of the basin in square miles

    Returns
    -------
    list
        indices of all peaks
    """
    # Compute the minimum distance between neighboring peaks
    # Find all local peaks with the minimum distance criterion of 5 + log(A)  where A is the basin area in square miles
    distance = 5 + int(round(np.log10(basin_area))) - 1
    peak_indices = []
    for i in range(distance, len(discharge) - distance):
        if discharge[i] > max(discharge[i-distance:i]) and discharge[i] > max(discharge[i+1:i+distance+1]):
            peak_indices.append(i)

    # Remove peaks that violate the minimum discharge criterion
    modified = True
    while modified:
        modified = False
        for i in range(len(peak_indices) - 1):
            min_discharge = min(discharge[peak_indices[i]:peak_indices[i+1]])
            min_peak = min([discharge[peak_indices[i]], discharge[peak_indices[i+1]]])
            if min_discharge >= 0.75 * min_peak:
                if discharge[peak_indices[i]] < discharge[peak_indices[i+1]]:
                    peak_indices.pop(i)
                else:
                    peak_indices.pop(i+1)
                modified = True
                break

    return peak_indices

# def get_mask(
#         data:pd.DataFrame,
#         target_name: str,
#         threshold: float,
#         extremes_type: Literal["high", "low"] = 'high',
#         r: int = 5,
#     ) -> np.array:
#     """
#     Create mask for tigramite, masking out all non-extreme values
#     Get extreme events from time series using the Peaks Over Threshold method.
#     Masking only target as in target_name.

#     Parameters
#     ----------
#     data : pandas.DataFrame
#         Time series input variables.
#     target_name: str
#         Name of the target column in data
#     extremes_type : str
#         high - get extreme high values (above threshold)
#         low - get extreme low values (below threshold)
#     threshold : float
#         Threshold used to find exceedances.
#     r : int duration of window used to decluster the exceedances in timesteps

#     Returns
#     -------
#     mask : np.array
#         array of mask, same shape as data

#     """
#     extremes = get_extremes_peaks_over_threshold(data[target_name],extremes_type=extremes_type,threshold=threshold, r = r)
#     mask = np.full(data.shape, False)
#     mask[:,data.columns.get_loc(target_name)] = [False if i in extremes.index else True for i in data.index]
#     return mask

def moving_average(data:np.array,smoothing:int) ->np.array:
    """calculates moving average with the returned array having the same lentgth as the input

    this is done by 'ramping up' the moving average, so that the first values are only partially moving averages

    Parameters
    ----------
    data : np.array
        input data, 1d
    smoothing : int
        moving average of timeseries with same length as input
    """
    smoothed = np.array([])
    for i in range(0,smoothing-1):
        smoothed = np.append(smoothed,data[0:i+1].mean())
    smoothed = np.append(smoothed,np.convolve(data,np.ones(smoothing),'valid')/smoothing )
    return(smoothed)