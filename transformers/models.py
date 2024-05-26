import re
import pandas as pd
import numpy as np
import logging

from sklearn.base import BaseEstimator, ClassifierMixin
import statsmodels.api as sm

from transformers.const import BIN, WOE


logger = logging.getLogger()


class CustomLogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    
    """
    Fit LogReg model (statsmodels)
    
    Input features detected by regex, for ex.:
     - fr"^({NUM}|{CAT})\w+"  -  to filter features that start "num__" and "cat__"
     - fr"\w+__bin$"  -   to filter features that end with "__bin"
    """
    
    p_value_max = 0.05

    def __init__(self, features_pattern = fr"\w+{BIN}{WOE}$", features_lst=None): 
        self.estimator = None  
        self.features_lst = features_lst
        self.features_pattern = re.compile(features_pattern)

    def fit(self, X, y):
        
        if self.features_lst:
            
            logger.info(f"Modeling: LogisticRegression with fixed set of features.")
            
            self.estimator = sm.Logit(
                y, 
                sm.add_constant(X[self.features_lst])
            ).fit(disp=0)

        else:
            
            logger.info(f"Modeling: LogisticRegression with feature selection.")
            
            self.features_lst = list(
                filter(
                    lambda x: re.match(self.features_pattern, x), X.columns
                )
            )

            while True:

                model = sm.Logit(
                    y,
                    sm.add_constant(X[self.features_lst])
                ).fit(disp=0)    
                
                model_stats = model.summary2().tables[1]
                
                worst = model_stats['P>|z|'].idxmax()
                worst_p = model_stats.loc[worst]['P>|z|']
                
                if worst_p > self.p_value_max:
                    
                    model_stats = model_stats.drop(worst)
                    self.features_lst.remove(worst)
                
                    logger.info(f"==========================")
                    logger.info(f"Features cnt: {len(self.features_lst)}")
                    logger.info(f"Dropped:      {worst}")
                    logger.info(f"P-value:      {worst_p:0.4}")
                    
                else:
                    break

        logger.info(f"Modeling: LogisticRegression - features amount:{len(self.features_lst)}")
        logger.info(f"Modeling: LogisticRegression - fit done.")
        logger.debug(model_stats)
        
        self.estimator = model
        return self

    def predict(self, X):
         
        prediction = self.estimator.predict(
            sm.add_constant(X[self.features_lst], has_constant='add')
        )
        
        logger.info(f"Modeling: LogisticRegression - prediction done.")
        
        return prediction