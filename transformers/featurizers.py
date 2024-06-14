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
        self.num_features_lst = num_features_lst
        self.cat_features_lst = cat_features_lst

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        
        X = X.copy()
        
        for f in self.num_features_lst:
            if (f in X.columns) and (NUM + f not in X.columns):
                X.loc[X[f].astype(str).isin(["", "nan"]), f] = np.nan
                X[NUM + f] = pd.to_numeric(X[f], errors='coerce').fillna(self.num_features_lst[f]).astype(float)

        for f in self.cat_features_lst:
            if (f in X.columns) and (CAT + f not in X.columns):
                X[CAT + f] = X[f].astype(str)
                X.loc[X[CAT + f].isin(["", "nan"]), CAT + f] = self.cat_features_lst[f]

        logger.info(f"Simple features - added.")
        
        if y is not None:
            return pd.concat([X, y], axis=1)
        else:
            return X
        
        
class FeatureEliminationTransformer(BaseEstimator, TransformerMixin):
    """
    Dropping:
        - Duplicated features
        - Constant and quasi constant
        
    Input features detected by regex, for ex.:
     - fr"^({NUM}|{CAT})\w+"  -  to filter features that start "num__" and "cat__"
     - fr"\w+__bin__woe$"  -   to filter features that end with "__bin"
    """
    
    def __init__(self, features_pattern=fr"^({NUM}|{CAT})\w+", constant_share_thr=0.98):

        self.constant_share_thr = constant_share_thr
        self.features_pattern = re.compile(features_pattern)
        self.features_to_drop = set()

    def fit(self, X, y):
        
        self.features_set = set(
            filter(
                lambda x: re.match(
                    self.features_pattern, x
                    ), X.columns
            
                )
            )
            
        logger.info(f"Feature elimination - initial features count:      {len(self.features_set)}")

        # 1. Drop duplicated features
        self.selector_dup = DropDuplicateFeatures(variables=list(self.features_set))
        self.selector_dup.fit(X)
        self.features_set = self.features_set - set(self.selector_dup.features_to_drop_)
        logger.info(f"Feature elimination - after dups dropping:         {len(self.features_set)}")
        
        # 2. Drop quazi-constant features       
        self.selector_const = DropConstantFeatures(variables=list(self.features_set), tol=self.constant_share_thr)
        self.selector_const.fit(X)
        self.features_set = self.features_set - set(self.selector_const.features_to_drop_)
        
        logger.info(f"Feature elimination - after constants dropping:     {len(self.features_set)}")
        logger.info(f"Feature elimination - fit done.")
        
        return self

    def transform(self, X, y=None):

        features_set = set(
            filter(
                lambda x: re.match(
                    self.features_pattern, x
                    ), X.columns
                )
            )

        X = X.drop(list(features_set - self.features_set), axis=1)

        logger.info(f"Feature elimination - transform done.")
        
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
