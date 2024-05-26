import pandas as pd
import numpy as np
from optbinning import OptimalBinning

from transformers.const import *


def get_values_map(input_map):
    output_map = {}
   
    for key_set, value in input_map.items():
        for v in key_set:
            output_map[v] = value
    
    return output_map


def get_optbin_info_cat(data, feature, target=TARGET, max_n_bins=4, min_bin_size=0.10, min_target_diff=0.02):
    
    x = data[feature].fillna(MISSING).values.astype(str)
    y = data[target].values

    optb = OptimalBinning(    
        dtype="categorical", 
        solver="cp", 
        prebinning_method="cart",
        min_event_rate_diff=min_target_diff, # minimal difference in event rate
        divergence='iv',                     # objective metric to maximize
        min_bin_size=min_bin_size,           # minimal fraction for bin size
        max_n_bins=max_n_bins,
        time_limit=10,
        min_prebin_size=0.01,
        max_n_prebins=50
    )
    
    optb.fit(x, y)
    groups_map_dct = {frozenset(split): GROUPS[i] for i, split in enumerate(optb.splits)}

    return groups_map_dct


def get_optbin_info_num(data, feature, target=TARGET, max_n_bins=4, min_bin_size=0.09, min_target_diff=0.02):
    
    x = pd.to_numeric(data[feature], errors='ignore').astype(float).fillna(NAN)
    y = data[target].values

    optb = OptimalBinning(    
        dtype="numerical", 
        solver="cp", 
        prebinning_method="cart",
        min_event_rate_diff=min_target_diff, # minimal difference in event rate
        divergence='iv',                     # objective metric to maximize
        min_bin_size=min_bin_size,           # minimal fraction for bin size
        time_limit=10,
        min_prebin_size=0.01,
        max_n_prebins=50,
        max_n_bins=max_n_bins,
    )
    
    optb.fit(x, y)
    bins_lst = [x.min()] + list(optb.splits) + [np.inf]

    return bins_lst
