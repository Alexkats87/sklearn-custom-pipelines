"""Encoding transformers for feature transformation."""

import re
import pandas as pd
import logging

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn_custom_pipelines.utils.const import (
    BIN, WOE, MISSING, OTHER, TARGET
)
from sklearn_custom_pipelines.utils.helpers import (
    get_optbin_info_cat,
    get_optbin_info_num,
    get_values_map,
    calculate_woe
)
from sklearn_custom_pipelines.utils.custom_mappings import features_custom_mappings_dct

logger = logging.getLogger(__name__)


class WoeEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Transform binned categorical features into Weight of Evidence (WOE) values.

    Uses custom WOE calculation algorithm:
    WOE = ln(% of events / % of non-events)

    Parameters
    ----------
    zero_filler : float, default=0.01
        Value to use when filling zeros to avoid log(0)
    """

    def __init__(self, zero_filler=0.01):
        self.zero_filler = zero_filler
        self.features_lst = None
        self.features_woe_lst = None
        self.woe_dict = None

    def fit(self, X, y):
        """Fit the WOE encoder."""
        self.features_lst = list(
            filter(lambda x: re.match(".*__bin$", x), X.columns)
        )

        self.features_woe_lst = [f + WOE for f in self.features_lst]
        self.woe_dict = {}

        # Calculate WOE for each binned feature
        for feature in self.features_lst:
            self.woe_dict[feature] = calculate_woe(
                X, y, feature, zero_filler=self.zero_filler
            )

        logger.info(f"WOE encoder - fit done.")

        return self

    def transform(self, X, y=None):
        """Transform features using WOE encoding."""
        X = X.copy()
        
        # Store WOE columns
        woe_cols = []
        
        # Convert binned features to string for mapping
        for feature in self.features_lst:
            if feature in X.columns:
                X[feature] = X[feature].astype(str)
                
                # Create WOE column
                woe_feature = feature + WOE
                X[woe_feature] = X[feature].map(self.woe_dict[feature])
                
                # Fill missing values with worst (max) WOE value
                if X[woe_feature].isna().any():
                    max_woe = max(self.woe_dict[feature].values())
                    X[woe_feature] = X[woe_feature].fillna(max_woe)
                
                # Ensure WOE column is numeric
                X[woe_feature] = pd.to_numeric(X[woe_feature], errors='coerce')
                woe_cols.append(woe_feature)
        
        # Drop the original binned columns (keep only WOE encoded)
        X = X.drop(columns=self.features_lst)

        logger.info(f"WOE encoder - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class BinningCategoriesTransformer(BaseEstimator, TransformerMixin):
    """
    Apply optimal binning to categorical features.

    Uses OptimalBinning to group categorical features based on
    information value and target distribution.

    Parameters
    ----------
    max_n_bins : int, default=4
        Maximum number of bins
    min_bin_size : float, default=0.10
        Minimum bin size as fraction of data
    min_target_diff : float, default=0.02
        Minimum target rate difference between bins
    """

    def __init__(self, max_n_bins=4, min_bin_size=0.10, min_target_diff=0.02):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.min_target_diff = min_target_diff
        self.binning_results_dct = None
        self.features_lst = None

    def fit(self, X, y):
        """Fit the binning transformer."""
        X = X.copy()

        self.features_lst = list(
            filter(lambda x: re.match(r"^(cat__)", x), X.columns)
        )
        X[self.features_lst] = X[self.features_lst].astype(str)

        self.binning_results_dct = {}

        for f in self.features_lst:
            try:
                bins_map_dct = get_optbin_info_cat(
                    data=pd.concat([X, y], axis=1),
                    feature=f,
                    target=TARGET,
                    max_n_bins=self.max_n_bins,
                    min_bin_size=self.min_bin_size,
                    min_target_diff=self.min_target_diff
                )
                self.binning_results_dct[f] = bins_map_dct
                logger.debug(
                    f"Processed: {f:50} , bins: {len(bins_map_dct.keys())}"
                )

            except Exception as e:
                logger.error(f"Cat. binning - can't fit {f}: {e}.")

        logger.info(f"Cat. binning - fit done.")
        return self

    def transform(self, X, y=None):
        """Apply binning to categorical features."""
        X = X.copy()

        for f in self.features_lst:
            if f in X.columns:
                try:
                    X[f + BIN] = (
                        X[f].astype(str)
                        .map(get_values_map(self.binning_results_dct[f]))
                        .fillna(MISSING)
                    )

                except Exception as e:
                    logger.error(f"Cat. binning - can't transform {f}: {e}.")

        logger.info(f"Cat. binning - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class BinningNumericalTransformer(BaseEstimator, TransformerMixin):
    """
    Apply optimal binning to numerical features.

    Uses OptimalBinning to discretize numerical features based on
    information value and target distribution.

    Parameters
    ----------
    max_n_bins : int, default=4
        Maximum number of bins
    min_bin_size : float, default=0.09
        Minimum bin size as fraction of data
    min_target_diff : float, default=0.02
        Minimum target rate difference between bins
    """

    def __init__(self, max_n_bins=4, min_bin_size=0.09, min_target_diff=0.02):
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.min_target_diff = min_target_diff
        self.binning_results_dct = None
        self.features_lst = None

    def fit(self, X, y):
        """Fit the binning transformer."""
        X = X.copy()

        self.features_lst = list(
            filter(lambda x: re.match(r"^(num__)", x), X.columns)
        )
        X[self.features_lst] = X[self.features_lst].fillna(-999.0).astype(float)

        self.binning_results_dct = {}

        for f in self.features_lst:
            try:
                bins_lst = get_optbin_info_num(
                    data=pd.concat([X, y], axis=1),
                    feature=f,
                    target=TARGET,
                    max_n_bins=self.max_n_bins,
                    min_bin_size=self.min_bin_size,
                    min_target_diff=self.min_target_diff
                )

                self.binning_results_dct[f] = bins_lst
                logger.debug(f"Processed: {f:50} , bins: {len(bins_lst) - 1}")

            except Exception as e:
                logger.error(f"Num. binning - can't fit {f}: {e}.")

        logger.info(f"Num. binning - fit done.")
        return self

    def transform(self, X, y=None):
        """Apply binning to numerical features."""
        X = X.copy()

        X[self.features_lst] = X[self.features_lst].fillna(-999.0).astype(float)

        for f in self.features_lst:
            if f in X.columns:
                try:
                    X[f + BIN] = pd.cut(
                        X[f],
                        self.binning_results_dct[f],
                        precision=0,
                        include_lowest=True,
                        right=False
                    ).astype(str).fillna(MISSING)

                    X[f + BIN] = X[f + BIN].astype(str)

                except Exception as e:
                    logger.error(f"Num. binning - can't transform {f}: {e}.")

        logger.info(f"Num. binning - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    Apply custom mappings to specified features.

    Uses predefined custom mappings from custom_mappings module.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self

    def transform(self, X, y=None):
        """Apply custom mappings to features."""
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
    Encode rare categorical values.

    Groups rare categories (those with frequency below threshold)
    into a single "Others" category.

    Parameters
    ----------
    tol : float, default=0.001
        Frequency threshold for rare categories
    n_categories : int, default=4
        Maximum number of categories to keep
    fill_na : str, default=MISSING
        Value to fill missing values
    replace_with : str, default=OTHER
        Value to replace rare categories with
    """

    def __init__(
        self,
        tol=0.001,
        n_categories=4,
        fill_na=MISSING,
        replace_with=OTHER
    ):
        self.tol = tol
        self.n_categories = n_categories
        self.cat_features_lst = None
        self.fill_na = fill_na
        self.replace_with = replace_with
        self.encoder_dict_ = {}

    def fit(self, X, y=None):
        """Fit the transformer and identify rare categories."""
        self.cat_features_lst = list(
            filter(lambda x: re.match(r"^(cat__)", x), X.columns)
        )
        self.encoder_dict_ = {}

        for f in self.cat_features_lst:
            if len(X[f].unique()) > self.n_categories:

                logger.debug(
                    f"Rare categories - process {f} with {len(X[f].unique())} categories."
                )

                # Learn the most frequent categories
                t = X[f].fillna(self.fill_na).astype(str).value_counts(normalize=True)

                # Non-rare labels
                freq_idx = t[t >= self.tol].index
                self.encoder_dict_[f] = list(freq_idx)

            else:
                self.encoder_dict_[f] = list(X[f].unique())

        logger.info(f"Rare categories - fit done")

        return self

    def transform(self, X, y=None):
        """Replace rare categories with OTHER."""
        X = X.copy()

        for f in self.cat_features_lst:
            if f in X.columns:
                X[f] = X[f].fillna(self.fill_na).astype(str)
                X.loc[~X[f].isin(self.encoder_dict_[f]), f] = self.replace_with

        logger.info(f"Rare categories - transform done")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X

