"""Feature transformation and selection transformers."""

import re
import pandas as pd
import numpy as np
import logging
from itertools import combinations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from feature_engine.selection import (
    DropDuplicateFeatures,
    DropConstantFeatures,
    DropCorrelatedFeatures,
    SelectByTargetMeanPerformance
)

from sklearn_custom_pipelines.utils.const import CAT, NUM, BIN, WOE, SEP
from sklearn_custom_pipelines.utils.helpers import get_values_map, get_optbin_info_cat

logger = logging.getLogger(__name__)


class SimpleFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Add new columns with prefixes "cat__" and "num__" for specified columns.
    
    These prefixed columns will be treated as initial features on the next
    steps of the pipeline.
    
    Parameters
    ----------
    num_features_missings_dct : dict
        Dictionary mapping numerical feature names to fill values for missing values
    cat_features_missings_dct : dict
        Dictionary mapping categorical feature names to fill values for missing values
    """

    def __init__(self, num_features_missings_dct, cat_features_missings_dct):
        self.num_features_missings_dct = num_features_missings_dct
        self.cat_features_missings_dct = cat_features_missings_dct

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        return self

    def transform(self, X, y=None):
        """Add prefixed feature columns to the dataframe."""
        X = X.copy()

        # Add numerical features with NUM prefix
        for f in self.num_features_missings_dct.keys():
            if (f in X.columns) and (NUM + f not in X.columns):
                X[NUM + f] = X[f].replace("", np.nan).fillna(
                    self.num_features_missings_dct[f]
                ).astype(float)
            else:
                X[NUM + f] = float(self.num_features_missings_dct[f])

        # Add categorical features with CAT prefix
        for f in self.cat_features_missings_dct.keys():
            if (f in X.columns) and (CAT + f not in X.columns):
                X[CAT + f] = X[f].fillna(self.cat_features_missings_dct[f]).astype(str)
            else:
                X[CAT + f] = self.cat_features_missings_dct[f]

        logger.info(f"Simple features - added.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class FeatureEliminationTransformer(BaseEstimator, TransformerMixin):
    """
    Drop features based on multiple criteria:
    - Duplicated features
    - Constant and quasi-constant features
    - Low information value (IV)
    - Highly correlated features (keeping the best performer)

    Based on Feature-engine library.

    Parameters
    ----------
    correlation_thr : float, default=0.8
        Correlation threshold for detecting correlated features
    constant_share_thr : float, default=0.98
        Threshold for constant/quasi-constant features
    iv_min : float, default=0.02
        Minimum information value threshold
    corr_features_selector_bins : int, default=5
        Number of bins for correlation feature selector
    corr_features_selector_strategy : str, default='equal_width'
        Strategy for binning in correlation selector
    features_pattern : str, optional
        Regex pattern to filter features to eliminate
    cat_features_pattern : str, default=r".*__bin$"
        Regex pattern for categorical features
    num_features_pattern : str, default=r"^(num__)"
        Regex pattern for numerical features
    """

    def __init__(
        self,
        correlation_thr=0.8,
        constant_share_thr=0.98,
        corr_features_selector_bins=5,
        iv_min=0.02,
        corr_features_selector_strategy='equal_width',
        features_pattern=None,
        cat_features_pattern=r".*__bin$",
        num_features_pattern=r"^(num__)"
    ):
        self.cat_features_pattern = cat_features_pattern
        self.num_features_pattern = num_features_pattern
        self.features_pattern = features_pattern
        self.correlation_thr = correlation_thr
        self.constant_share_thr = constant_share_thr
        self.iv_min = iv_min
        self.corr_features_selector_bins = corr_features_selector_bins
        self.corr_features_selector_strategy = corr_features_selector_strategy
        self.features_to_drop = set()

    @staticmethod
    def _calculate_iv(X, y, columns):
        """Calculate the Information Value (IV) for multiple features."""
        iv_dict = {}
        for feature in columns:
            df = pd.concat([X[feature], y], axis=1)
            df.columns = [feature, 'target']

            total_bad = df['target'].sum()
            total_good = df['target'].count() - total_bad

            grouped = df.groupby(feature).agg({'target': ['sum', 'count']})
            grouped.columns = ['bad', 'total']
            grouped['good'] = grouped['total'] - grouped['bad']

            grouped['bad_dist'] = grouped['bad'] / total_bad
            grouped['good_dist'] = grouped['good'] / total_good
            grouped['woe'] = np.log(
                (grouped['good_dist'] + 1e-9) / (grouped['bad_dist'] + 1e-9)
            )
            grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']

            iv_dict[feature] = grouped['iv'].sum()

        return iv_dict

    def fit(self, X, y):
        """Fit the transformer and identify features to drop."""
        X = X.copy()
        self.features_to_drop = set()

        self.cat_features_set = set(
            filter(lambda x: re.match(self.cat_features_pattern, x), X.columns)
        )
        self.num_features_set = set(
            filter(lambda x: re.match(self.num_features_pattern, x), X.columns)
        )

        if self.features_pattern:
            pattern_features = set(
                filter(lambda x: re.match(self.features_pattern, x), X.columns)
            )
            self.all_features_set = pattern_features
        else:
            self.all_features_set = self.cat_features_set.union(self.num_features_set)

        logger.info(
            f"Feature elimination - initial features count: {len(self.all_features_set)}"
        )

        # 1. Drop duplicated features
        self.selector_dup = DropDuplicateFeatures(variables=list(self.all_features_set))
        self.selector_dup.fit(X)
        dropped_dups = set(self.selector_dup.features_to_drop_)
        self.all_features_set = self.all_features_set - dropped_dups
        self.features_to_drop = self.features_to_drop | dropped_dups
        logger.info(f"Feature elimination - after dups dropping: {len(self.all_features_set)}")

        # 2. Drop quasi-constant features
        self.selector_const = DropConstantFeatures(
            variables=list(self.all_features_set), tol=self.constant_share_thr
        )
        self.selector_const.fit(X)
        dropped_const = set(self.selector_const.features_to_drop_)
        self.all_features_set = self.all_features_set - dropped_const
        self.features_to_drop = self.features_to_drop | dropped_const
        logger.info(f"Feature elimination - after constants dropping: {len(self.all_features_set)}")

        # 3. Filter by IV (information value) - only for categorical features
        cat_features_in_set = self.cat_features_set.intersection(self.all_features_set)
        if len(cat_features_in_set) > 0:
            iv_dict = self._calculate_iv(X, y, list(cat_features_in_set))
            iv_features_to_drop = [k for k, v in iv_dict.items() if ((v < self.iv_min) or (v > 0.45))]
            self.all_features_set = self.all_features_set - set(iv_features_to_drop)
            self.features_to_drop = self.features_to_drop | set(iv_features_to_drop)
            logger.info(f"Feature elimination - after IV filter: {len(self.all_features_set)}")

        # 4. Detect and remove correlated features - only for numerical features
        num_features_in_set = self.num_features_set.intersection(self.all_features_set)
        if len(num_features_in_set) > 0:
            self.selector_corr = DropCorrelatedFeatures(
                variables=list(num_features_in_set), threshold=self.correlation_thr
            )
            self.selector_corr.fit(X)
            correlated_feature_sets = self.selector_corr.correlated_feature_sets_
            logger.info(
                f"Feature elimination - groups of corr features: {len(correlated_feature_sets)}"
            )

            # 5. Select best features from groups using target performance
            for feature_set in correlated_feature_sets:
                feature_lst = list(feature_set)
                logger.debug(f"Group size: {len(feature_lst)}")

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
                logger.debug(f"Best: {feature_best}")

                self.all_features_set = self.all_features_set - set(feature_lst)
                self.features_to_drop = self.features_to_drop | set(feature_lst)

        logger.info(f"Feature elimination - final features count: {len(self.all_features_set)}")
        logger.info(f"Feature elimination - fit done.")
        return self

    def transform(self, X, y=None):
        """Remove identified features from the dataframe."""
        X = X.copy()
        X = X.drop(list(self.features_to_drop), axis=1, errors='ignore')

        logger.info(f"Feature elimination - dropped features count: {len(self.features_to_drop)}")
        logger.info(f"Feature elimination - selected features count: {len(self.all_features_set)}")
        logger.info(f"Feature elimination - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class DecorrelationTransformer(BaseEstimator, TransformerMixin):
    """
    Detect and remove correlated features.

    Based on Feature-engine library. Detects groups of correlated features
    based on correlation threshold and selects one feature from each group
    based on mean target performance.

    Parameters
    ----------
    correlation_thr : float, default=0.8
        Correlation threshold for detecting correlated features
    features_pattern : str, optional
        Regex pattern to filter features
    """

    def __init__(self, correlation_thr=0.8, features_pattern=None):
        self.correlation_thr = correlation_thr
        self.features_pattern = features_pattern
        self.features_to_drop = set()

    def fit(self, X, y):
        """Fit the transformer and identify correlated features to drop."""
        X = X.copy()

        if self.features_pattern:
            features = list(
                filter(lambda x: re.match(self.features_pattern, x), X.columns)
            )
        else:
            features = X.columns.tolist()

        self.selector_corr = DropCorrelatedFeatures(
            variables=features, threshold=self.correlation_thr
        )
        self.selector_corr.fit(X)

        self.features_to_drop = set(self.selector_corr.features_to_drop_)
        logger.info(
            f"Decorrelation - detected correlated features: {len(self.features_to_drop)}"
        )
        logger.info(f"Decorrelation - fit done.")

        return self

    def transform(self, X, y=None):
        """Remove correlated features from the dataframe."""
        X = X.copy()
        X = X.drop(list(self.features_to_drop), axis=1, errors='ignore')

        logger.info(f"Decorrelation - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class PairedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Create paired features from high-cardinality categorical features.

    Combines categorical features to create interaction features
    and selects pairs based on information value.

    Parameters
    ----------
    features_pattern : str, default=r".*__bin$"
        Regex pattern to identify categorical features
    max_cardinality : int, default=6
        Maximum cardinality for features to pair
    iv_min : float, default=0.01
        Minimum information value threshold
    iv_max : float, default=0.5
        Maximum information value threshold
    """

    def __init__(
        self,
        features_pattern=r".*__bin$",
        max_cardinality=6,
        iv_min=0.01,
        iv_max=0.5
    ):
        self.features_pattern = re.compile(features_pattern)
        self.max_cardinality = max_cardinality
        self.features_to_pair = None
        self.iv_min = iv_min
        self.iv_max = iv_max
        self.features_pairs_lst = []

    @staticmethod
    def _calculate_iv(X, y, feature):
        """Calculate the Information Value (IV) for a feature."""
        df = pd.concat([X[feature], y], axis=1)
        df.columns = [feature, 'target']

        total_bad = df['target'].sum()
        total_good = df['target'].count() - total_bad

        grouped = df.groupby(feature).agg({'target': ['sum', 'count']})
        grouped.columns = ['bad', 'total']
        grouped['good'] = grouped['total'] - grouped['bad']

        grouped['bad_dist'] = grouped['bad'] / total_bad
        grouped['good_dist'] = grouped['good'] / total_good
        grouped['woe'] = np.log(
            (grouped['good_dist'] + 1e-9) / (grouped['bad_dist'] + 1e-9)
        )
        grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']

        return grouped['iv'].sum()

    def fit(self, X, y=None):
        """Fit the transformer and identify good feature pairs."""
        self.cat_features = list(
            filter(lambda x: re.match(self.features_pattern.pattern, x), X.columns)
        )

        # Select candidates to pair by cardinality
        cardinalities = X[self.cat_features].nunique()
        features_to_pair_lst = cardinalities[
            cardinalities.between(2, self.max_cardinality)
        ].index.tolist()

        # List of pairs
        all_features_pairs_lst = list(combinations(features_to_pair_lst, 2))

        logger.info(f"Paired features fit: {len(features_to_pair_lst)} features to pair.")
        logger.info(
            f"Paired features fit: {len(all_features_pairs_lst)} candidates for pairs."
        )

        self.features_pairs_lst = []

        for p in all_features_pairs_lst:
            f_paired_name = SEP.join(p)
            X_temp = X.copy()
            X_temp[f_paired_name] = X_temp[p[0]].astype(str) + SEP + X_temp[p[1]].astype(str)

            iv = self._calculate_iv(X_temp, y, f_paired_name)

            if self.iv_min < iv < self.iv_max:
                self.features_pairs_lst.append(p)
                logger.debug(f"Paired features fit: {f_paired_name} to add with IV={iv:0.3}")

        logger.info(f"Paired features fit: {len(self.features_pairs_lst)} pairs to add.")

        return self

    def transform(self, X, y=None):
        """Add paired features to the dataframe."""
        X = X.copy()

        for p in self.features_pairs_lst:
            f_paired_name = SEP.join(p)
            if (p[0] in X.columns) and (p[1] in X.columns):
                X[f_paired_name] = X[p[0]].astype(str) + SEP + X[p[1]].astype(str)

        logger.info(f"Paired features: {len(self.features_pairs_lst)} added.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class CustomPCATransformer(BaseEstimator, TransformerMixin):
    """
    Apply PCA transformation to numerical features.

    Applies standard scaling followed by PCA to numerical features
    (identified by "num__" prefix).

    Parameters
    ----------
    n_components : int, optional
        Number of PCA components to keep
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.num_features_lst = None

    def fit(self, X, y=None):
        """Fit the scaler and PCA to numerical features."""
        self.num_features_lst = list(
            filter(lambda x: re.match(r"^(num__)", x), X.columns)
        )

        X_selected = X[self.num_features_lst]
        X_selected = self.scaler.fit_transform(X_selected)

        self.pca.fit(X_selected)
        logger.info(f"PCA - top var ratios: {self.pca.explained_variance_ratio_[:6]}")
        logger.info(f"PCA - total var ratio: {sum(self.pca.explained_variance_ratio_):5.4}")

        return self

    def transform(self, X, y=None):
        """Apply PCA transformation to numerical features."""
        X = X.copy()

        X_selected = X[self.num_features_lst]
        X_selected = self.scaler.transform(X_selected)
        X_other = X.drop(columns=self.num_features_lst)

        # Apply PCA
        X_pca = self.pca.transform(X_selected)
        pca_comp_names = [f'num__pca{i+1}' for i in range(self.n_components)]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_comp_names, index=X.index)

        # Combine PCA-transformed data with other columns
        X_combined = pd.concat([X_other, X_pca_df], axis=1)

        if y is not None:
            return pd.concat([X_combined, y], axis=1)
        else:
            return X_combined
