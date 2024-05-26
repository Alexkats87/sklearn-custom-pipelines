import re
import pandas as pd
import numpy as np
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.encoding import WoEEncoder
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
    """
    Transforming categirical features/bins into weight of evidence (WOE)
    """
    
    def __init__(self, zero_filler=0.01):
        self.zero_filler = zero_filler
        self.worst_bins_dct = {}
        self.fitting = False
        
    def _to_str(self, df):
        df[self.features_lst] = df[self.features_lst].astype(str)
        return df

    def fit(self, X, y):
        
        self.fitting = True

        self.features_lst = list(
            filter(
                lambda x: re.match(f".*{BIN}$", x), X.columns
            )
        )
        X = self._to_str(X)

        self.woe_encoder = WoEEncoder(
            variables=self.features_lst
        ).fit(
            X[self.features_lst],
            y
        )
        
        # Get the worst bin for every feature: highest WOE corresponds to worst bin
        # We fill features without bins by the worst bin during inference
        for f in self.features_lst:         
            self.worst_bins_dct[f] = max(
                self.woe_encoder.encoder_dict_[f],
                key=lambda k: self.woe_encoder.encoder_dict_[f][k]
            )
            
        
        logger.info(f"WOE cat encoder - fit done.")
        
        return self

    def transform(self, X, y=None):
        
        X = X.copy()
        X = self._to_str(X)
        
        if not self.fitting:
        
            features_with_missings = (X[X!='nan'])[self.features_lst].isnull().sum()
            features_with_missings = features_with_missings[features_with_missings > 0]
            
            for f in features_with_missings.index:
                X.loc[X[f] == 'nan', f] = self.worst_bins_dct[f]
                
        self.fitting = False
        
        X_orig = X[self.features_lst].copy()
        
        X = self.woe_encoder.transform(X[self.features_lst])
        X = X.rename(
            columns={c: c + WOE for c in self.features_lst}
        )
        
        X = pd.concat([X, X_orig], axis=1)

        logger.info(f"WOE cat encoder - transform done.")
                
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    Applying custom mappings to categorical features
    """
    
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
    """
    Encoding rare values of categorical features into one value "Others"
    to reduce cardinality
    """
    
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

        self.rare_encoder.fit(X)
        logger.info(f"Rare categories encoder - fit done.")
        
        return self

    def transform(self, X, y=None):
        
        X = X.copy()
        
        X[self.cat_features_lst] = X[self.cat_features_lst].astype(str)
        X = self.rare_encoder.transform(X)

        logger.info(f"Rare categories encoder - transform done.")
                
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
      
        
class BinningCategoriesTransformer(BaseEstimator, TransformerMixin):
    """
    Binning for categorical features
    """
    
    def __init__(self, max_n_bins=4, min_bin_size=0.10, min_target_diff=0.02):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.min_target_diff = min_target_diff
        self.binning_results_dct = None

    def fit(self, X, y):

        X = X.copy()
        
        self.features_lst = list(filter(lambda x: re.match(f"^({CAT})", x), X.columns))
        X[self.features_lst] = X[self.features_lst].astype(str)

        self.binning_results_dct = {}

        for f in self.features_lst:
        
            bins_map_dct = get_optbin_info_cat(
                data=pd.concat([X, y], axis=1), 
                feature=f, 
                target=TARGET
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
    """
    Binning for numerical features
    """
    
    def __init__(self, max_n_bins=4, min_bin_size=0.09, min_target_diff=0.02):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.min_target_diff = min_target_diff
        self.binning_results_dct = None

    def fit(self, X, y):
        
        X = X.copy()

        self.features_lst = list(filter(lambda x: re.match(f"^({NUM})", x), X.columns))
        X[self.features_lst] = X[self.features_lst].fillna(NAN).astype(float)

        self.binning_results_dct = {}

        for f in self.features_lst:
        
            bins_lst = get_optbin_info_num(
                data=pd.concat([X, y], axis=1), 
                feature=f, 
                target=TARGET
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
        
        
