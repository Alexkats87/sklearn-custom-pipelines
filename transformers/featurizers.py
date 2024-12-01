import re
import pandas as pd
import numpy as np
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.selection import (
    DropDuplicateFeatures, 
    DropConstantFeatures, 
    DropCorrelatedFeatures, 
    SelectByTargetMeanPerformance
)

from transformers.const import CAT, NUM, BIN, WOE


logger = logging.getLogger()


class SimpleFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Adding to initial pandas DataFrame `X` new columns with prefixes "cat__" and "num__" for specified columns, 
    that will be treated as initial features on the next steps of pipeline
    """
    
    def __init__(self, num_features_lst, cat_features_lst):
        self.num_features_missings_dct = num_features_missings_dct
        self.cat_features_missings_dct = cat_features_missings_dct

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        X = X.copy()
        
        for f in self.num_features_missings_dct.keys():
            if (f in X.columns) and (NUM + f not in X.columns):
                X[NUM + f] = X[f].replace("", np.nan).fillna(self.num_features_missings_dct[f]).astype(float)
            else:
                X[NUM + f] = float(self.num_features_missings_dct[f])

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
    Dropping:
        - Duplicated features
        - Constant and quasi constant features
        - From highly-correlated features groups selected the most powerful
        - IV-based (information value) feature selection

    """
    
    def __init__(
        self, 
        correlation_thr=0.8, 
        constant_share_thr=0.98, 
        corr_features_selector_bins=5, 
        iv_min=0.02, 
        corr_features_selector_strategy='equal_width',
        cat_features_pattern = r".*__bin$",
        num_features_pattern = r"^(num__)"
        
    ):
        self.cat_features_pattern = cat_features_pattern
        self.num_features_pattern = num_features_pattern
        self.correlation_thr = correlation_thr
        self.constant_share_thr = constant_share_thr
        self.iv_min = iv_min
        self.corr_features_selector_bins = corr_features_selector_bins
        self.corr_features_selector_strategy = corr_features_selector_strategy
        self.features_to_drop = set()

    @staticmethod
    def _calculate_iv(X, y, columns):
        """
        Calculate the Information Value (IV) for multiple features.
        
        Parameters:
        X (pd.DataFrame): The input dataframe with features.
        y (pd.Series): The target variable series.
        columns (list): A list of columns for which to calculate IV.
        
        Returns:
        dict: A dictionary with columns as keys and their IV as values.
        """
        iv_dict = {}
        for feature in columns:
            # Combine X[feature] and y into a DataFrame for processing
            df = pd.concat([X[feature], y], axis=1)
            df.columns = [feature, 'target']
    
            # Calculate the distribution of bad and good (target 1 and 0)
            total_bad = df['target'].sum()
            total_good = df['target'].count() - total_bad
    
            # Group by feature and calculate the WoE and IV
            grouped = df.groupby(feature).agg({'target': ['sum', 'count']})
            grouped.columns = ['bad', 'total']
            grouped['good'] = grouped['total'] - grouped['bad']
    
            # To avoid division by zero, add a small constant to good and bad
            grouped['bad_dist'] = grouped['bad'] / total_bad
            grouped['good_dist'] = grouped['good'] / total_good
            grouped['woe'] = np.log((grouped['good_dist'] + 1e-9) / (grouped['bad_dist'] + 1e-9))
            grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']
    
            # Sum IV for the feature
            iv_dict[feature] = grouped['iv'].sum()
        
        return iv_dict

    def fit(self, X, y):
        X = X.copy()
        self.features_to_drop = set()

        self.cat_features_set = set(filter(lambda x: re.match(self.cat_features_pattern, x), X.columns))
        self.num_features_set = set(filter(lambda x: re.match(self.num_features_pattern, x), X.columns))
        self.all_features_set = self.cat_features_set.union(self.num_features_set)
        logger.info(f"Feature elimination - initial features count:      {len(self.all_features_set)}")

        # 1. Drop duplicated features
        self.selector_dup = DropDuplicateFeatures(variables=list(self.all_features_set))
        self.selector_dup.fit(X)
        self.cat_features_set = self.cat_features_set - set(self.selector_dup.features_to_drop_)
        self.num_features_set = self.num_features_set - set(self.selector_dup.features_to_drop_)
        self.all_features_set = self.all_features_set - set(self.selector_dup.features_to_drop_)
        self.features_to_drop = self.features_to_drop | set(self.selector_dup.features_to_drop_)
        logger.info(f"Feature elimination - after dups dropping:         {len(self.all_features_set)}")
        
        # 2. Drop quazi-constant features       
        self.selector_const = DropConstantFeatures(variables=list(self.all_features_set), tol=self.constant_share_thr)
        self.selector_const.fit(X)
        self.cat_features_set = self.cat_features_set - set(self.selector_const.features_to_drop_)
        self.num_features_set = self.num_features_set - set(self.selector_const.features_to_drop_)
        self.all_features_set = self.all_features_set - set(self.selector_const.features_to_drop_)
        self.features_to_drop = self.features_to_drop | set(self.selector_const.features_to_drop_)
        logger.info(f"Feature elimination - after constants dropping:     {len(self.all_features_set)}")

        # 3. Filter by IV (information value)
        if len(self.cat_features_set) > 0:
            iv_dict = self._calculate_iv(X, y, list(self.cat_features_set))
            iv_features_to_drop = [k for k,v in iv_dict.items() if ((v < self.iv_min) or (v > 0.45))]
            self.cat_features_set = self.cat_features_set - set(iv_features_to_drop)
            self.all_features_set = self.all_features_set - set(iv_features_to_drop)
            self.features_to_drop = self.features_to_drop | set(iv_features_to_drop)
            logger.info(f"Feature elimination - after IV filter:     {len(self.all_features_set)}")
        

        # 4. Detect groups of corr features
        if len(self.num_features_set) > 0:
            self.selector_corr = DropCorrelatedFeatures(variables=list(self.num_features_set), threshold=self.correlation_thr)   
            self.selector_corr.fit(X)
            correlated_feature_sets = self.selector_corr.correlated_feature_sets_
            logger.info(f"Feature elimination - groups of corr features:      {len(correlated_feature_sets)}")
    
            # 5. Select best features from groups of corr. features using SelectByTargetMeanPerformance class
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
                
                self.num_features_set = self.num_features_set - set(feature_lst)
                self.all_features_set = self.all_features_set - set(feature_lst)
                self.features_to_drop = self.features_to_drop | set(feature_lst)

        logger.info(f"Feature elimination - initial features count:      {len(self.all_features_set)}.")
        logger.info(f"Feature elimination - fit done.")
        return self

    def transform(self, X, y=None):
        X = X.copy()

        X = X.drop(list(self.features_to_drop), axis=1, errors='ignore')

        logger.info(f"Feature elimination - dropped  features count:   {len(self.features_to_drop)}.")
        logger.info(f"Feature elimination - selected features count:   {len(self.all_features_set)}.")
        logger.info(f"Feature elimination - transform done.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
