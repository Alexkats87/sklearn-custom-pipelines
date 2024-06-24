import re
import pandas as pd
import numpy as np
import logging

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import statsmodels.api as sm

from transformers.const import BIN, WOE, CAT, NUM


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
        
        model_stats = None
        
        if self.features_lst:
            logger.info(f"Modeling: LogisticRegression with fixed set of features.")
            logger.debug(f" {self.features_lst}.")
            self.estimator = sm.Logit(
                y, 
                sm.add_constant(X[self.features_lst])
            ).fit(disp=0)
            
            model_stats = self.estimator.summary2().tables[1]

        else:
            
            logger.info(f"Modeling: LogisticRegression with feature selection.")
            
            self.features_lst = list(
                filter(
                    lambda x: re.match(self.features_pattern, x), X.columns
                )
            )

            while True:

                self.estimator = sm.Logit(
                    y,
                    sm.add_constant(X[self.features_lst])
                ).fit(disp=0)    
                
                model_stats = self.estimator.summary2().tables[1]
                
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

        return self

    def predict(self, X):
         
        prediction = self.estimator.predict(
            sm.add_constant(X[self.features_lst], has_constant='add')
        ).to_list()
        
        # For testing: returns full dataset with prediction column
        # if self.return_full_dataset:
        #     X['y_pred'] = prediction
        #     return X
        
        # For test and batch prediction : returns full dataset with prediction column
        if X.shape[0] > 1:
            X['y_pred'] = prediction
            return X
        
        # For prod and single prediction: composes output JSON
        features_bins_list = [f[:-5] for f in self.features_lst]
        
        output = {}
        output['WOE'] = X[self.features_lst].round(3).to_dict(orient='records')[0]
        output['Bins'] = X[features_bins_list].to_dict(orient='records')[0]
        output['Score'] = round(prediction[0], 5)
        
        logger.info(f"Modeling: LogisticRegression - prediction done.")
        
        return output
    
    
class CustomCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """    
    Fit CatBoostClassifier model, perform iterative feature selection
    excluding features with low feature importance after every model fit.
    
    Uses evaluation set for early stopping.
    
    Input features detected by prefixes: 
    - "num__" for numerical
    - "cat__" for categorical
    
    """
    
    importance_min = 0.01

    def __init__(self, **params):
        
        self.params = params
        self.estimator = CatBoostClassifier(**self.params)
            
        self.cat_features_set = None
        self.num_features_set = None

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self.estimator, param, value)
        return self

    def fit(self, X, y, **params):

        self.cat_features_set = set(filter(lambda x: re.match(fr"^({CAT})", x), X.columns))
        self.num_features_set = set(filter(lambda x: re.match(fr"^({NUM})", x), X.columns))

        logger.info(f"Modeling: Catboost - Initial features amount: {len(self.cat_features_set | self.num_features_set)}")

        iter = 1
        while True:
            
            self.estimator.fit(
                X=X[list(self.cat_features_set | self.num_features_set)],
                y=y,
                cat_features=list(self.cat_features_set),
                eval_set=(
                    params['eval_set'][0][list(self.cat_features_set | self.num_features_set)],
                    params['eval_set'][1]
                ),
                verbose=0,
            )
            
            importance_dct = dict(zip(
                self.estimator.feature_names_, 
                self.estimator.get_feature_importance()
            ))
            
            features_to_drop_set = set([key for key, value in importance_dct.items() if value <= self.importance_min])
            
            if not features_to_drop_set:           
                logger.info(f"Modeling: Catboost - Selected features amount: {len(self.cat_features_set | self.num_features_set)}")               
                break
            else:
                self.cat_features_set = self.cat_features_set - set(features_to_drop_set)
                self.num_features_set = self.num_features_set - set(features_to_drop_set)
                logger.debug(f"==========================")
                logger.debug(f"Iter. {iter}, dropped features amt: {len(features_to_drop_set)}")

            iter += 1

        iterations_count = self.estimator.get_best_iteration() + 1
        logger.debug(f"Iterations_count: {iterations_count}")

        
        self.estimator.fit(
            X=X[list(self.cat_features_set | self.num_features_set)],
            y=y,
            cat_features=list(self.cat_features_set),
            eval_set=(
                params['eval_set'][0][list(self.cat_features_set | self.num_features_set)],
                params['eval_set'][1]
            ),
            verbose=0,
        )

        logger.info(f"Modeling: Catboost - fit done.")

        return self

    def predict(self, X):
        
        prediction = self.estimator.predict_proba(X[list(self.cat_features_set | self.num_features_set)])
        logger.info(f"Modeling: Catboost - prediction done.")
        
        return prediction
