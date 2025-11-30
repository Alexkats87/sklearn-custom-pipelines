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
        

class ClippingTransformer(BaseEstimator, TransformerMixin):
    """
    Clip numerical features to specified quantile bounds and create binary clipping indicators.
    
    This transformer handles outliers by clipping numerical features to lower and upper quantile
    values learned during training. Additionally, it creates binary indicator features that mark
    which values were clipped, preserving information about outliers.
    
    Clipping is useful for:
    - Handling extreme outliers that could skew model performance
    - Reducing the impact of heavy-tailed distributions
    - Stabilizing model predictions on new data with potential outliers
    - Creating interpretable outlier indicators for the model
    
    Parameters
    ----------
    num_features_pattern : str, default=r"^(num__)"
        Regex pattern to identify numerical features to clip.
        By default, matches features starting with "num__" prefix.
        Example: r"^(num__)" matches "num__age", "num__salary", etc.
    
    q_lower : float, default=0.005
        Lower quantile threshold for clipping (0.005 = 0.5th percentile).
        Values below this quantile are clipped to the quantile value.
        Must be between 0 and 1.
    
    q_upper : float, default=0.995
        Upper quantile threshold for clipping (0.995 = 99.5th percentile).
        Values above this quantile are clipped to the quantile value.
        Must be between 0 and 1.
    
    create_clipped_flag : bool, default=True
        Whether to create binary indicator columns for clipped values.
        If True, for each clipped feature, two new columns are created:
        - 'num__clipped_low__<feature>': 1 if value was clipped at lower bound
        - 'num__clipped_high__<feature>': 1 if value was clipped at upper bound
        If False, only the clipped values are returned without indicators.
    
    Attributes
    ----------
    num_features_lst : list
        List of numerical feature column names identified during fit()
    
    q_lower_dct : dict
        Dictionary mapping feature names to their lower quantile values computed during fit()
    
    q_upper_dct : dict
        Dictionary mapping feature names to their upper quantile values computed during fit()
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn_custom_pipelines import ClippingTransformer
    >>> 
    >>> # Create sample data with outliers
    >>> X = pd.DataFrame({
    ...     'num__age': [20, 25, 30, 150, 200, 35],  # 150, 200 are outliers
    ...     'num__salary': [30000, 50000, 45000, 40000, 1000000, 55000]  # 1000000 is outlier
    ... })
    >>> y = pd.Series([0, 1, 0, 1, 1, 0])
    >>> 
    >>> # Initialize and fit the transformer
    >>> transformer = ClippingTransformer(q_lower=0.1, q_upper=0.9)
    >>> transformer.fit(X, y)
    >>> 
    >>> # Transform the data
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed.head())
    
    Notes
    -----
    - Only features matching the pattern are clipped
    - If create_clipped_flag is True, for each clipped feature, two binary indicators are created:
      - 'num__clipped_low__<feature>': 1 if value was clipped at lower bound
      - 'num__clipped_high__<feature>': 1 if value was clipped at upper bound
    - If create_clipped_flag is False, only clipped values are returned
    - Clipping bounds are computed during fit() and applied during transform()
    - Non-matching features are passed through unchanged
    - The transformer is compatible with sklearn Pipeline
    - If y is provided to transform(), it is concatenated to the output
    
    See Also
    --------
    PowerNormTransformer : For power-based normalization of features
    """
    
    def __init__(
        self,
        num_features_pattern=r"^(num__)",
        q_lower=0.005,
        q_upper=0.995,
        create_clipped_flag=True
    ):
        """
        Initialize ClippingTransformer.
        
        Parameters
        ----------
        num_features_pattern : str, default=r"^(num__)"
            Regex pattern for identifying numerical features to clip
        q_lower : float, default=0.005
            Lower quantile threshold (must be between 0 and 1)
        q_upper : float, default=0.995
            Upper quantile threshold (must be between 0 and 1)
        create_clipped_flag : bool, default=True
            Whether to create binary indicator columns showing which values were clipped.
            If True, creates 'num__clipped_low__<feature>' and 'num__clipped_high__<feature>' columns.
            If False, only returns clipped feature values without indicators.
        """
        self.num_features_pattern = num_features_pattern
        self.q_lower = q_lower
        self.q_upper = q_upper
        self.create_clipped_flag = create_clipped_flag
        self.num_features_lst = None
        self.q_lower_dct = None
        self.q_upper_dct = None

    def fit(self, X, y=None):
        """
        Fit the ClippingTransformer to the data.
        
        Identifies numerical features matching the pattern and computes the lower and upper
        quantile values for each feature that will be used for clipping.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features dataframe. Must contain at least one column matching
            the num_features_pattern
        
        y : pd.Series or None, default=None
            Target variable. Not used for fitting the transformer, but included
            for sklearn Pipeline compatibility
        
        Returns
        -------
        self : ClippingTransformer
            Returns self for method chaining
        """
        X = X.copy()
        
        self.num_features_lst = list(filter(lambda x: re.match(self.num_features_pattern, x), X.columns))
        self.q_lower_dct = {}
        self.q_upper_dct = {}

        for f in self.num_features_lst:
            self.q_lower_dct[f] = X[f].quantile(self.q_lower)
            self.q_upper_dct[f] = X[f].quantile(self.q_upper)

        logger.info(f"Clipping - fit done.")
        
        return self

    def transform(self, X, y=None):
        """
        Apply clipping and optionally create binary indicators for clipped values.
        
        Clips features to the quantile bounds learned during fit(). If create_clipped_flag
        is True, creates binary indicators showing which values were clipped.
        Non-matching features are passed through unchanged.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features dataframe to transform. Must have the same columns
            (or at least the same numerical features) as used during fit()
        
        y : pd.Series or None, default=None
            Target variable. If provided, it will be concatenated to the output
        
        Returns
        -------
        X_transformed : pd.DataFrame
            Dataframe with clipped numerical features and optional binary indicator columns.
            Original feature values are replaced with clipped values.
            If create_clipped_flag is True, for each clipped feature 'num__feature',
            two new columns are added:
            - 'num__clipped_low__feature': Binary flag (1 if clipped at lower bound)
            - 'num__clipped_high__feature': Binary flag (1 if clipped at upper bound)
            If y is provided, it is appended as an additional column.
        
        Notes
        -----
        - Original feature values are replaced with clipped values
        - Values below lower quantile are clipped to the lower quantile value
        - Values above upper quantile are clipped to the upper quantile value
        - Binary indicators help preserve information about outliers for the model
        - If create_clipped_flag is False, no indicator columns are created
        """
        X = X.copy()
        
        for f in self.num_features_lst:
            
            X[f] = X[f].clip(
                lower=self.q_lower_dct[f], 
                upper=self.q_upper_dct[f]
            )
            
            if self.create_clipped_flag:
                X[NUM + 'clipped_low__' + f] = 0
                X[NUM + 'clipped_high__' + f] = 0
                X.loc[X[f] <= self.q_lower_dct[f],  NUM + 'clipped_low__' + f] = 1
                X.loc[X[f] >= self.q_upper_dct[f],  NUM + 'clipped_high__' + f] = 1

        logger.info(f"Clipping - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
