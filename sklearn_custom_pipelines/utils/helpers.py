"""Helper functions for sklearn-custom-pipelines."""

import pandas as pd
import numpy as np
from optbinning import OptimalBinning

from sklearn_custom_pipelines.utils.const import (
    MISSING, TARGET, NAN
)


def get_values_map(input_map):
    """
    Convert a mapping of frozensets to a flat dictionary.
    
    Parameters
    ----------
    input_map : dict
        Dictionary with frozenset keys and value mappings
        
    Returns
    -------
    dict
        Flattened mapping dictionary
    """
    output_map = {}
    for key_set, value in input_map.items():
        for v in key_set:
            output_map[v] = value
    return output_map


def get_optbin_info_cat(
    data,
    feature,
    target=TARGET,
    max_n_bins=4,
    min_bin_size=0.10,
    min_target_diff=0.02
):
    """
    Calculate optimal binning for categorical features using OptimalBinning.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing feature and target
    feature : str
        Feature column name
    target : str, default='y'
        Target column name
    max_n_bins : int, default=4
        Maximum number of bins
    min_bin_size : float, default=0.10
        Minimum bin size as fraction
    min_target_diff : float, default=0.02
        Minimum target rate difference
        
    Returns
    -------
    dict
        Dictionary mapping frozensets of categories to bin groups
    """
    x = data[feature].fillna(MISSING).values.astype(str)
    y = data[target].values

    optb = OptimalBinning(
        dtype="categorical",
        solver="cp",
        prebinning_method="cart",
        min_event_rate_diff=min_target_diff,
        divergence='iv',
        min_bin_size=min_bin_size,
        max_n_bins=max_n_bins,
        time_limit=10,
        min_prebin_size=0.01,
        max_n_prebins=50
    )

    optb.fit(x, y)
    groups_map_dct = {
        frozenset(split): chr(65 + i)  # A, B, C, ...
        for i, split in enumerate(optb.splits)
    }

    return groups_map_dct


def get_optbin_info_num(
    data,
    feature,
    target=TARGET,
    max_n_bins=4,
    min_bin_size=0.09,
    min_target_diff=0.02
):
    """
    Calculate optimal binning for numerical features using OptimalBinning.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing feature and target
    feature : str
        Feature column name
    target : str, default='y'
        Target column name
    max_n_bins : int, default=4
        Maximum number of bins
    min_bin_size : float, default=0.09
        Minimum bin size as fraction
    min_target_diff : float, default=0.02
        Minimum target rate difference
        
    Returns
    -------
    list
        List of bin edges for pd.cut function
    """
    x = pd.to_numeric(data[feature], errors='coerce').astype(float).fillna(NAN)
    y = data[target].values

    optb = OptimalBinning(
        dtype="numerical",
        solver="cp",
        prebinning_method="cart",
        min_event_rate_diff=min_target_diff,
        divergence='iv',
        min_bin_size=min_bin_size,
        time_limit=10,
        min_prebin_size=0.01,
        max_n_prebins=50,
        max_n_bins=max_n_bins,
    )

    optb.fit(x, y)
    bins_lst = [x.min()] + list(optb.splits) + [np.inf]

    return bins_lst


def calculate_woe(X, y, feature, zero_filler=0.01):
    """
    Calculate Weight of Evidence (WOE) for a categorical feature.
    
    WOE = ln(% of events / % of non-events)
    
    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe
    y : pd.Series
        Target variable (binary: 0/1 or False/True)
    feature : str
        Feature column name to calculate WOE for
    zero_filler : float, default=0.01
        Value to fill zeros to avoid log(0)
        
    Returns
    -------
    dict
        Dictionary mapping category values to their WOE values
    """
    # Convert y to numeric array
    y_vals = np.asarray(y).astype(int).flatten()
    x_vals = np.asarray(X[feature]).astype(str)
    
    # Get total events and non-events
    total_events = (y_vals == 1).sum()
    total_non_events = (y_vals == 0).sum()
    
    # Calculate WOE for each category
    woe_dict = {}
    for category in np.unique(x_vals):
        mask = x_vals == category
        cat_events = (y_vals[mask] == 1).sum()
        cat_non_events = (y_vals[mask] == 0).sum()
        
        # Calculate percentages with zero_filler
        pct_events = max(cat_events / total_events, zero_filler) if total_events > 0 else zero_filler
        pct_non_events = max(cat_non_events / total_non_events, zero_filler) if total_non_events > 0 else zero_filler
        
        # Calculate WOE
        woe_dict[category] = np.log(pct_events / pct_non_events)
    
    return woe_dict
