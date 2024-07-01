import re
import pandas as pd
# import numpy as np
import logging

from sklearn.base import BaseEstimator, TransformerMixin

from feature_engine.encoding import WoEEncoder
from feature_engine.selection import (
    DropCorrelatedFeatures, 
    SelectByTargetMeanPerformance
)
from feature_engine.encoding.rare_label import RareLabelEncoder

from transformers.helpers import (
    get_optbin_info_cat, 
    get_optbin_info_num, 
    get_values_map
)

from transformers.custom_mappings import features_custom_mappings_dct
from transformers.const import *


logger = logging.getLogger()


class WoeEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, zero_filler=0.01):
        self.zero_filler = zero_filler
        self.worst_bins_dct = {}
        
    def _to_str(self, df):
        df[self.features_lst] = df[self.features_lst].astype(str)
        return df

    def fit(self, X, y):
        self.features_lst = list(
            filter(
                lambda x: re.match(".*__bin$", x), X.columns
            )
        )
        
        self.features_woe_lst = [f + WOE for f in self.features_lst]
            
        X = self._to_str(X)

        self.woe_encoder = WoEEncoder(
            variables=self.features_lst
        ).fit(
            X[self.features_lst],
            y
        )
        
        logger.info(f"WOE cat encoder - fit done.")
        
        return self

    def transform(self, X, y=None):
        
        X = X.copy()
        X = self._to_str(X)   
        X_orig = X[self.features_lst].copy()
        
        X = self.woe_encoder.transform(X[self.features_lst])
        X = X.rename(
            columns={c: c + WOE for c in self.features_lst}
        )
        
        X = pd.concat([X, X_orig], axis=1)

        # Get list of features with missings
        woe_features_with_missings = X[self.features_woe_lst].isna().sum()
        woe_features_with_missings = woe_features_with_missings[woe_features_with_missings > 0]
        
        # Fill missings (if any) by worst value, i.e. by max WOE
        for f_woe in woe_features_with_missings.index:
            f_bin = f_woe[:-5]
            X.loc[X[f_woe].isna(), f_woe] = max(self.woe_encoder.encoder_dict_[f_bin].values())
        
        logger.info(f"WOE cat encoder - transform done.")
                
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
        
        
class BinningCategoriesTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_n_bins=4, min_bin_size=0.10, min_target_diff=0.02):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.min_target_diff = min_target_diff
        self.binning_results_dct = None

    def fit(self, X, y):

        X = X.copy()
        
        self.features_lst = list(filter(lambda x: re.match(r"^(cat__)", x), X.columns))
        X[self.features_lst] = X[self.features_lst].astype(str)

        self.binning_results_dct = {}

        for f in self.features_lst:
        
            bins_map_dct = get_optbin_info_cat(
                data=pd.concat([X, y], axis=1), 
                feature=f, 
                target=TARGET,
                max_n_bins = self.max_n_bins,
                min_bin_size = self.min_bin_size,
                min_target_diff = self.min_target_diff
            )
     
            self.binning_results_dct[f] = bins_map_dct
            logger.debug(f"Processed: {f:50} , bins: {len(bins_map_dct.keys())}")

        logger.info(f"Cat. binning - fit done.")
        return self

    def transform(self, X, y=None):
        
        X = X.copy()

        for f in self.features_lst:
            if f in X.columns:
                
                X[f + BIN] = (
                    X[f].astype(str)
                        .map(get_values_map(self.binning_results_dct[f]))
                        .fillna(MISSING)
                )
        
        logger.info(f"Cat. binning - tranfsorm done.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X        


class BinningNumericalTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_n_bins=4, min_bin_size=0.09, min_target_diff=0.02):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.min_target_diff = min_target_diff
        self.binning_results_dct = None

    def fit(self, X, y):
        
        X = X.copy()

        self.features_lst = list(filter(lambda x: re.match(r"^(num__)", x), X.columns))
        X[self.features_lst] = X[self.features_lst].fillna(NAN).astype(float)

        self.binning_results_dct = {}

        for f in self.features_lst:
        
            bins_lst = get_optbin_info_num(
                data=pd.concat([X, y], axis=1), 
                feature=f, 
                target=TARGET,
                max_n_bins = self.max_n_bins,
                min_bin_size = self.min_bin_size,
                min_target_diff = self.min_target_diff
            )
              
            self.binning_results_dct[f] = bins_lst
            logger.debug(f"Processed: {f:50} , bins: {len(bins_lst) - 1}")

        logger.info(f"Num. binning - fit done.")
        return self

    def transform(self, X, y=None):
        
        X = X.copy()

        X[self.features_lst] = X[self.features_lst].fillna(NAN).astype(float)

        for f in self.features_lst:
            if f in X.columns:
                
                X[f + BIN] = pd.cut(
                    X[f], 
                    self.binning_results_dct[f], 
                    precision=0, 
                    include_lowest=True, 
                    right=False
                ).astype(str).fillna(MISSING)
                
                X[f + BIN] = X[f + BIN].astype(str)
        
        logger.info(f"Num. binning - tranfsorm done.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
        
        
class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        
        X = X.copy()

        for f, m in features_custom_mappings_dct.items():
            if f in X.columns:
                X[f] = X[f].map(get_values_map(m)).fillna(MISSING)
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class RareCategoriesTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, tol=0.01, n_categories=4):
        self.tol = tol
        self.n_categories = n_categories
        self.cat_features_lst = None

    def fit(self, X, y=None):

        X = X.copy()
        
        self.cat_features_lst = list(
            filter(
                lambda x: re.match(f"^({CAT})", x), X.columns
            )
        )
        
        X[self.cat_features_lst] = X[self.cat_features_lst].astype(str)

        self.rare_encoder = RareLabelEncoder(
            tol=self.tol,
            n_categories=self.n_categories,
            replace_with=OTHER,
            variables=self.cat_features_lst,
        )

        self.rare_encoder.fit(X[self.cat_features_lst])
        logger.info(f"Rare categories encoder - fit done.")
        
        return self

    def transform(self, X, y=None):
        
        X = X.copy()
        
        X[self.cat_features_lst] = X[self.cat_features_lst].astype(str)
        X[self.cat_features_lst] = self.rare_encoder.transform(X[self.cat_features_lst])
        
        logger.info(f"Rare categories encoder - transform done.")
                
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
        
        
class DecorrelationTransformer(BaseEstimator, TransformerMixin):
    """
    Detection and removing correlated features:
        - Detect groups of correlated features
        - From every group only one feature is selected based on mean target performance.
          Other features will be dropped
          
    Input features detected by regex, for ex.:
     - fr"^({NUM}|{CAT})\w+"  -  to filter features that start "num__" and "cat__"
     - fr"\w+__bin$"  -   to filter features that end with "__bin"
    
    """
    
    def __init__(
        self, 
        correlation_thr=0.8, 
        corr_features_selector_bins=5, 
        corr_features_selector_strategy='equal_width',
        features_pattern = fr"\w+{BIN}{WOE}$"
    ):
        self.correlation_thr = correlation_thr
        self.corr_features_selector_bins = corr_features_selector_bins
        self.corr_features_selector_strategy = corr_features_selector_strategy
        self.features_pattern = re.compile(features_pattern)
        self.features_to_drop = set()

    def fit(self, X, y):

        self.features_set = set(
            filter(
                lambda x: re.match(self.features_pattern, x), X.columns
            )
        )
        
        if len(self.features_set) <=0:
            raise ValueError("Feature decorrelation - input features set is empty")
        
        logger.info(f"Feature decorrelation - initial features count:      {len(self.features_set)}.")

        # Detect groups of corr features
        self.selector_corr = DropCorrelatedFeatures(variables=list(self.features_set), threshold=self.correlation_thr)   
        self.selector_corr.fit(X)
        correlated_feature_sets = self.selector_corr.correlated_feature_sets_
        logger.info(f"Feature decorrelation - groups of corr features:      {len(correlated_feature_sets)}")

        # Select best features from groups of corr. features
        for feature_set in correlated_feature_sets:
            
            feature_lst = list(feature_set)

            logger.debug(f"Group size:  {len(feature_lst)}")
            logger.debug(f"Features:    {feature_lst}")
            
            mean_target_selector = SelectByTargetMeanPerformance(
                bins=self.corr_features_selector_bins, 
                strategy=self.corr_features_selector_strategy
            )
            mean_target_selector.fit(X[feature_lst], y)
            
            feature_best = max(
                mean_target_selector.feature_performance_, 
                key=lambda k: mean_target_selector.feature_performance_[k]
            )
            
            feature_lst.remove(feature_best)          

            logger.debug(f"Best:        {feature_best}")
            
            self.features_set = self.features_set - set(feature_lst)

        logger.info(f"Feature decorrelation - features count after:      {len(self.features_set)}.")
        logger.info(f"Feature decorrelation - fit done.")
        return self

    def transform(self, X, y=None):
        
        

        features_set = set(
            filter(
                lambda x: re.match(self.features_pattern, x), X.columns
            )
        )

        X = X.drop(list(features_set - self.features_set), axis=1)
        
        logger.info(f"Feature decorrelation - transform done.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
