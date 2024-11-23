import re
import pandas as pd
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


class CustomRareCategoriesTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, tol=0.001, n_categories=4, fill_na=MISSING, replace_with=OTHER):
        self.tol = tol
        self.n_categories = n_categories
        self.cat_features_lst = None
        self.fill_na = fill_na
        self.replace_with = replace_with

    def fit(self, X, y=None):

        self.cat_features_lst = list(filter(lambda x: re.match(r"^(cat__)", x), X.columns))
        self.encoder_dict_ = {}

        for f in self.cat_features_lst:
            if len(X[f].unique()) > self.n_categories:

                logger.debug(f"Rare categories encoder - process {f} with {len(X[f].unique())} unique categories.")

                # if the variable has more than the indicated number of categories
                # the encoder will learn the most frequent categories
                t = X[f].fillna(self.fill_na).astype(str).value_counts(normalize=True)

                # non-rare labels:
                freq_idx = t[t >= self.tol].index
                self.encoder_dict_[f] = list(freq_idx)

            else:
                self.encoder_dict_[f] = list(X[f].unique())
        
        logger.info(f"Rare categories encoder - fit done")
        
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for f in self.cat_features_lst:
            if f in X.columns:
                X[f] = X[f].fillna(self.fill_na).astype(str)
                X.loc[~X[f].isin(self.encoder_dict_[f]), f] = (self.replace_with)

        logger.info(f"Rare categories encoder - transform done")
                
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class PairedFeaturesTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, features_pattern=r"^cat__*__bin$", max_cardinality=6):
        self.features_pattern = re.compile(features_pattern)
        self.max_cardinality = max_cardinality
        self.features_to_pair = None

    def fit(self, X, y=None):
        
        self.cat_features = list(
            filter(
                lambda x: re.match(
                    self.features_pattern, x
                    ), X.columns
                )
            )

        cardinalities = X[self.cat_features].nunique()
        self.features_to_pair = cardinalities[cardinalities.between(2, self.max_cardinality)].index.to_list()

        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        features_to_pair = [c for c in self.features_to_pair if c in X.columns]

        features_pairs = (
            list(combinations(features_to_pair, 2))
        )

        cnt = 0

        for p in features_pairs:
        
            f_paired_name = SEP.join(p)
            X[f_paired_name] = X[list(p)].apply(lambda row: SEP.join(row.values.astype(str)), axis=1)
            cnt += 1

        logger.info(f"Paired features: {cnt} - added.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class CustomPCATransformer(BaseEstimator, TransformerMixin):
    """
        Custom PCA combined with standard scaler. 
        Considers features to transform by "num__" prefix
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):

        self.num_features_lst = list(filter(lambda x: re.match(r"^(num__)", x), X.columns))

        X_selected = X[self.num_features_lst]
        X_selected = self.scaler.fit_transform(X_selected)

        self.pca.fit(X_selected)
        logger.info(f"PCA - top var ratios: {self.pca.explained_variance_ratio_[:6]}")
        logger.info(f"PCA - fit done, total var ratio: {sum(self.pca.explained_variance_ratio_):5.4}")

        return self

    def transform(self, X, y=None):

        X = X.copy()
        
        X_selected = X[self.num_features_lst]
        X_selected = self.scaler.transform(X_selected)
        X_other = X.drop(columns=self.num_features_lst)  # Other columns remain untouched     

        # Apply PCA
        X_pca = self.pca.transform(X_selected)
        pca_comp_names = [f'num__pca{i+1}' for i in range(self.n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_comp_names, index=X.index)

        # Combine PCA-transformed data with the untouched columns
        X_combined = pd.concat([X_other, X_pca_df], axis=1)

        if y is not None:
            return pd.concat([X_combined, y], axis=1)
        else:
            return X_combined
