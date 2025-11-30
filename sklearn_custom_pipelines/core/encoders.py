"""Encoding transformers for feature transformation."""

import re
import pandas as pd
import numpy as np
import logging
import operator

from itertools import combinations, groupby
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from sklearn_custom_pipelines.utils.const import (
    BIN, WOE, MISSING, OTHER, TARGET, SEP
)
from sklearn_custom_pipelines.utils.helpers import (
    get_optbin_info_cat,
    get_optbin_info_num,
    get_values_map,
    calculate_woe,
    calculate_iv
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

    This transformer applies user-defined mappings to transform feature values
    based on a provided dictionary of mappings. Each feature can have a custom
    transformation rule that maps existing values to new values. Supports both
    one-to-one and many-to-one mappings via frozenset keys.

    Parameters
    ----------
    features_mappings_dct : dict
        Dictionary where keys are feature names and values are mapping definitions.
        Each mapping can be a dict or a list defining the transformation for that feature.
        Many-to-one mappings use frozenset keys to group multiple source values into
        a single target value.

    Examples
    --------
    Many-to-one mapping (group multiple values into categories):

    >>> from sklearn_custom_pipelines.utils.const import MISSING
    >>> education_map = {
    ...     frozenset({'Graduate', 'HND'}): 'Graduate',
    ...     frozenset({'Post Graduate'}): 'Post Graduate',
    ...     frozenset({'Primary', 'Secondary'}): 'Primary and Secondary',
    ...     frozenset({MISSING}): MISSING,
    ... }
    >>> mappings = {'education_status': education_map}
    >>> transformer = CustomMappingTransformer(features_mappings_dct=mappings)
    >>> transformer.fit(X, y)
    >>> X_transformed = transformer.transform(X)

    Notes
    -----
    - Unmapped values are filled with the MISSING constant
    - The transformer is a no-op during fit (stateless after initialization)
    - Transform is applied only to columns present in both the input and mappings dict
    - Many-to-one mappings use frozenset keys to map multiple input values to a single output
    """

    def __init__(self, features_mappings_dct=None):
        self.features_mappings_dct = features_mappings_dct if features_mappings_dct is not None else {}

    def fit(self, X, y=None):
        """Fit the transformer (no-op for this transformer)."""
        if self.features_mappings_dct:
            logger.debug(f"CustomMapping - fit called with {len(self.features_mappings_dct)} mappings")
        return self

    def transform(self, X, y=None):
        """Apply custom mappings to features."""
        X = X.copy()

        if not self.features_mappings_dct:
            logger.debug("CustomMapping - no mappings defined, skipping")
            if y is not None:
                return pd.concat([X, y], axis=1)
            else:
                return X

        for f, m in self.features_mappings_dct.items():
            if f in X.columns:
                values_map = get_values_map(m)
                
                # Count unmapped values before filling
                unmapped_count = X[f].map(values_map).isna().sum()
                
                # Apply mapping
                X[f] = X[f].map(values_map).fillna(MISSING)
                
                # Log results
                if unmapped_count > 0:
                    logger.info(f"CustomMapping - {f}: applied mapping, {unmapped_count} unmapped values filled with {MISSING}")
                else:
                    logger.info(f"CustomMapping - {f}: applied mapping successfully")
            else:
                logger.warning(f"CustomMapping - {f}: column not found in data, skipping")

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


class PairedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Create feature interactions by pairing binned categorical features.

    Generates combined features from pairs of binned categorical features by
    concatenating their values. Candidate pairs are selected based on their
    Information Value (IV) being within a specified range, ensuring meaningful
    interactions that have predictive power without being redundant.

    The transformer:
    1. Identifies features matching the pattern and having cardinality within limits
    2. Generates all possible pairs of such features
    3. Concatenates pair values (e.g., "bin1__SEP__bin2") and calculates IV
    4. Keeps only pairs with IV between iv_min and iv_max
    5. Applies selected pairs to new data during transform

    Parameters
    ----------
    features_pattern : str, default=r".*__bin$"
        Regex pattern to identify binned features to consider for pairing
        (e.g., features ending with '__bin')
    max_cardinality : int, default=6
        Maximum number of unique values a feature can have to be paired.
        Features with higher cardinality are excluded to reduce explosion
        of feature combinations
    iv_min : float, default=0.01
        Minimum Information Value threshold. Pairs with IV <= this value
        are excluded (likely too weak to be useful)
    iv_max : float, default=0.5
        Maximum Information Value threshold. Pairs with IV >= this value
        are excluded (likely redundant with existing features)

    Attributes
    ----------
    features_pairs_lst : list
        List of feature pairs selected during fit, stored as tuples of
        feature name pairs that will be combined
    cat_features : list
        List of all features matching the pattern found during fit
    """
    def __init__(self, features_pattern=r".*__bin$", max_cardinality=6, iv_min=0.01, iv_max=0.5):
        self.features_pattern = re.compile(features_pattern)
        self.max_cardinality = max_cardinality
        self.features_to_pair = None
        self.iv_min = iv_min
        self.iv_max = iv_max

    def fit(self, X, y=None):
        self.cat_features = list(
            filter(
                lambda x: re.match(
                    self.features_pattern, x
                    ), X.columns
                )
            )

        # Select candidates to pair by cardinality
        cardinalities = X[self.cat_features].nunique()
        features_to_pair_lst = cardinalities[cardinalities.between(2, self.max_cardinality)].index.to_list()

        # List of pairs
        all_features_pairs_lst = (
            list(combinations(features_to_pair_lst, 2))
        )

        logger.info(f"Paired features fit: {len(features_to_pair_lst)} features to pair.")
        logger.info(f"Paired features fit: {len(all_features_pairs_lst)} candidates for paired features to add.")

        cnt = 0

        self.features_pairs_lst = []

        for p in all_features_pairs_lst:
        
            f_paired_name = SEP.join(p)
            X[f_paired_name] = X[p[0]].astype(str) + SEP + X[p[1]].astype(str)

            iv = calculate_iv(X, y, f_paired_name)

            if self.iv_min < iv < self.iv_max:
                self.features_pairs_lst.append(p)
                logger.debug(f"Paired features fit: {f_paired_name} to add with IV={iv:0.3}")
            else:
                del X[f_paired_name]

            cnt += 1

        logger.info(f"Paired features fit: {len(self.features_pairs_lst)} paired features to add.")

        return self

    def transform(self, X, y=None):
        X = X.copy()

        cnt = 0

        for p in self.features_pairs_lst:
        
            f_paired_name = SEP.join(p)
            if (p[0] in X.columns) and (p[1] in X.columns):
                X[f_paired_name] = X[p[0]].astype(str) + SEP + X[p[1]].astype(str)
                cnt += 1

        logger.info(f"Paired features: {cnt} - added.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X


class PairedBinaryFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Create binary feature interactions using logical operations (OR, AND, XOR).

    Generates new features from pairs of binary flag features by applying logical
    operations (OR, AND, XOR). This transformer specifically works with binary
    features (containing only '0' and '1' values) and creates feature combinations
    that may capture complex interactions between flags.

    The transformer:
    1. Selects only binary/flag features (containing only '0' and '1')
    2. Identifies features matching the pattern (e.g., 'cat__flag__*')
    3. Generates all possible pairs and applies each logical operation
    4. Calculates Information Value for each operation result
    5. Keeps only combinations with IV between iv_min and iv_max
    6. Applies selected combinations during transform

    Parameters
    ----------
    features_pattern : str, default=r"cat__flag__"
        Regex pattern to identify binary flag features to consider for pairing
        (e.g., features starting with 'cat__flag__')
    iv_min : float, default=0.02
        Minimum Information Value threshold. Combinations with IV <= this value
        are excluded
    iv_max : float, default=0.5
        Maximum Information Value threshold. Combinations with IV >= this value
        are excluded

    Attributes
    ----------
    features_pairs_lst : list
        List of tuples containing (feature_pair, operation) that were selected
        during fit (e.g., (('feat1', 'feat2'), 'OR'))
    bin_features : list
        List of binary features matching the pattern found during fit
    OP_MAP : dict
        Mapping of operation names ('OR', 'AND', 'XOR') to operator functions
    """
    
    OP_MAP = {
        'OR': operator.or_, 
        'AND': operator.and_, 
        'XOR': operator.xor
    }
    
    def __init__(
        self, 
        features_pattern=r"cat__flag__",  
        iv_min=0.02, 
        iv_max=0.5
    ):
        self.features_pattern = re.compile(features_pattern)
        self.features_to_pair = None
        self.iv_min = iv_min
        self.iv_max = iv_max
        self.bin_features = None

    def fit(self, X, y=None):
        # TODO: remove?
        X = X.copy()

        # Select only binary/flag features
        binary_cols = X.columns[X.isin(['0', '1']).all()]
        X = X[binary_cols].copy()

        # Select only features which satisfy the pattern
        self.bin_features = list(
            filter(
                lambda x: re.match(
                    self.features_pattern, x
                    ), X.columns
                )
            )

        # List of pairs
        all_paired_features_lst = (
            list(combinations(self.bin_features, 2))
        )
        
        logger.info(f"Paired bin features fit: {len(self.bin_features)} features to pair.")
        logger.info(f"Paired bin features fit: {len(all_paired_features_lst)} candidates for paired features to add.")

        # Total added features count
        cnt = 0

        self.features_pairs_lst = []


        for op, op_func in self.OP_MAP.items():
        
            for p in all_paired_features_lst:
            
                f_paired_name = SEP.join(p) + SEP + op
                X[f_paired_name] = op_func(X[p[0]].astype(int) , X[p[1]].astype(int))
    
                iv = calculate_iv(X, y, f_paired_name)
    
                if self.iv_min < iv < self.iv_max:
                    self.features_pairs_lst.append((p, op))
                    logger.info(f"Paired features fit: {f_paired_name} to add with IV={iv:0.3}")
                else:
                    del X[f_paired_name]
    
                cnt += 1

        logger.info(f"Paired features fit: {len(self.features_pairs_lst)} paired features to add.")

        return self

    def transform(self, X, y=None):
        X = X.copy()

        cnt = 0

        for p, op in self.features_pairs_lst:  
            f_paired_name = SEP.join(p) + SEP + op       
            if (p[0] in X.columns) and (p[1] in X.columns):

                op_func = self.OP_MAP[op]
                
                X[f_paired_name] = op_func(X[p[0]].astype(int) , X[p[1]].astype(int)).astype(str)
                cnt += 1

        logger.info(f"Paired features: {cnt} - added.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
        
class PowerNormTransformer(BaseEstimator, TransformerMixin):
    """
    Apply power transformations to numerical features for normalization.
    
    Applies power transformations (Yeo-Johnson or Box-Cox) to numerical features
    to make them more Gaussian-like and improve model performance. The transformer
    automatically identifies numerical features based on a naming pattern and
    applies the specified power transformation method.
    
    This transformer is particularly useful for:
    - Normalizing skewed distributions
    - Stabilizing variance across different feature ranges
    - Improving model performance for algorithms sensitive to feature distributions
    - Preparing data for statistical tests that assume normality
    
    Parameters
    ----------
    num_features_pattern : str, default=r"^(num__)"
        Regex pattern to identify numerical features to transform.
        By default, matches features starting with "num__" prefix.
        Example: r"^(num__)" matches "num__age", "num__salary", etc.
    
    method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
        Power transformation method to use:
        - 'yeo-johnson': Works with any real-valued input, including zero and negative values
        - 'box-cox': Works only with strictly positive values
    
    Attributes
    ----------
    num_features_lst : list
        List of numerical feature column names identified during fit()
    
    power_tranformer : PowerTransformer
        Fitted sklearn PowerTransformer instance used for transformation
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn_custom_pipelines import PowerNormTransformer
    >>> 
    >>> # Create sample data with skewed distributions
    >>> X = pd.DataFrame({
    ...     'num__age': np.random.exponential(2, 100),
    ...     'num__income': np.random.exponential(5, 100),
    ...     'cat__city': ['NYC', 'LA', 'CHI'] * 33 + ['NYC']
    ... })
    >>> y = pd.Series(np.random.randint(0, 2, 100))
    >>> 
    >>> # Initialize and fit the transformer
    >>> transformer = PowerNormTransformer(method='yeo-johnson')
    >>> transformer.fit(X, y)
    >>> 
    >>> # Transform the data
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed.head())
    
    Notes
    -----
    - Only features matching the pattern are transformed
    - Non-matching features are passed through unchanged
    - The transformer is compatible with sklearn Pipeline
    - If y is provided to transform(), it is concatenated to the output
    
    See Also
    --------
    sklearn.preprocessing.PowerTransformer : The underlying sklearn transformer
    """

    def __init__(
        self,
        num_features_pattern=r"^(num__)",
        method='yeo-johnson'
    ):
        """
        Initialize PowerNormTransformer.
        
        Parameters
        ----------
        num_features_pattern : str, default=r"^(num__)"
            Regex pattern for identifying numerical features
        method : {'yeo-johnson', 'box-cox'}, default='yeo-johnson'
            Power transformation method
        """
        self.num_features_pattern = num_features_pattern
        self.method = method
        self.num_features_lst = None
        self.power_tranformer = None

    def fit(self, X, y=None):
        """
        Fit the PowerNormTransformer to the data.
        
        Identifies numerical features matching the pattern and fits a PowerTransformer
        to learn the optimal power transformation parameters from the training data.
        If no numerical features are found, the transformer is initialized but no fitting
        is performed on the PowerTransformer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features dataframe. Can contain any columns; if no columns match
            the num_features_pattern, fitting is skipped and data is passed through unchanged.
        
        y : pd.Series or None, default=None
            Target variable. Not used for fitting the transformer, but included
            for sklearn Pipeline compatibility
        
        Returns
        -------
        self : PowerNormTransformer
            Returns self for method chaining
        """
        X = X.copy()
        
        self.num_features_lst = list(filter(lambda x: re.match(self.num_features_pattern, x), X.columns))
        
        # Only fit PowerTransformer if there are numerical features to transform
        if len(self.num_features_lst) > 0:
            self.power_tranformer = PowerTransformer(method=self.method).fit(X[self.num_features_lst])
        else:
            self.power_tranformer = None

        logger.info(f"Power Transformer - fit done.")
        
        return self

    def transform(self, X, y=None):
        """
        Apply power transformation to numerical features.
        
        Applies the fitted power transformation to features identified during fit().
        Non-matching features are passed through unchanged. If no numerical features
        were found during fit(), returns the data unchanged.
        
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
            Dataframe with transformed numerical features. All columns from input
            are preserved, with numerical features replaced by their transformed
            values. If y is provided, it is appended as an additional column.
            If no numerical features were found, returns X unchanged.
        
        Notes
        -----
        - Columns matching num_features_lst are replaced with transformed values
        - Other columns are preserved as-is
        - If no numerical features exist, data passes through unchanged
        - If y is None, only features are returned
        - If y is not None, y is concatenated as an additional column
        """
        X = X.copy()
        
        # Only transform if numerical features were found during fit
        if len(self.num_features_lst) > 0 and self.power_tranformer is not None:
            X[self.num_features_lst] = self.power_tranformer.transform(X[self.num_features_lst])

        logger.info(f"Power Transformer - transform done.")

        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X