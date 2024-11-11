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

    def __init__(self, features_pattern = fr"\w+{BIN}{WOE}$"): 
        self.estimator = None  
        self.features_pattern = re.compile(features_pattern)
        self.model_stats = None

    def fit(self, X, y):
        
        model_stats = None

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
            
                logger.debug(f"==========================")
                logger.debug(f"Features cnt: {len(self.features_lst)}")
                logger.debug(f"Dropped:      {worst}")
                logger.debug(f"P-value:      {worst_p:0.4}")
                
            else:
                break

        logger.info(f"Modeling: LogisticRegression - features amount:{len(self.features_lst)}")
        logger.info(f"Modeling: LogisticRegression - fit done.")
        
        self.model_stats = model_stats
        logger.debug(model_stats)

        return self

    def predict(self, X):
         
        prediction = self.estimator.predict(
            sm.add_constant(X[self.features_lst], has_constant='add')
        ).to_list()

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
    
    """

    importance_thr = 0.03
    
    def __init__(self, **params):
        self.params = params
        self.estimator = CatBoostClassifier(**self.params)
        self.cat_features_set = None
        self.num_features_set = None

    def set_params(self, **params):
        self.params.update(params)
        self.estimator = CatBoostClassifier(**self.params)
        return self

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, **params):
        self.cat_features_set = set(filter(lambda x: re.match(r".*__bin$", x), X.columns))
        self.num_features_set = set(filter(lambda x: re.match(r"^(num__)", x), X.columns))

        logger.info(f"Modeling: Catboost - params: {self.estimator.get_params()}")
        logger.info(f"Modeling: Catboost - Initial features amount: {len(self.cat_features_set | self.num_features_set)}")

        X_valid = X.sample(frac=0.05, random_state=0)
        X_train = X.drop(X_valid.index)

        y_valid = y[X_valid.index]
        y_train = y[X_train.index]

        # Feature selection process
        iter = 1
        while True:

            self.estimator.fit(
                X=X_train[list(self.cat_features_set | self.num_features_set)],
                y=y_train,
                cat_features=list(self.cat_features_set),
                eval_set=(
                    X_valid[list(self.cat_features_set | self.num_features_set)],
                    y_valid
                ),
                early_stopping_rounds=early_stopping_rounds,
                verbose=0,
            )
            
            importance_dct = dict(zip(
                self.estimator.feature_names_,
                self.estimator.get_feature_importance()
            ))

            features_to_drop_set = set([key for key, value in importance_dct.items() if value <= 0.03])

            if not features_to_drop_set:
                logger.info(f"Modeling: Catboost - Selected features amount: {len(self.cat_features_set | self.num_features_set)}")
                break
            else:
                self.cat_features_set -= features_to_drop_set
                self.num_features_set -= features_to_drop_set
                logger.info(f"Iter. {iter}, dropped features amt: {len(features_to_drop_set)}")

            iter += 1

        try:
            iterations_count = self.estimator.get_best_iteration() + 1
            logger.info(f"Iters count: {iterations_count}")
        except Exception as e:
            logger.warning(f"Can't calc iters count: {e}")

        self.params['cat_features'] = list(self.cat_features_set)
        self.params['iterations'] = iterations_count
        logger.info(f"Modeling: Catboost on fit - params: {self.estimator.get_params()}")
        
        self.model_calibr = CalibratedClassifierCV(
            base_estimator=CatBoostClassifier(**self.params),
            method='sigmoid'
        )

        self.model_calibr.fit(
            X=X[list(self.cat_features_set | self.num_features_set)], 
            y=y
        )
        
        logger.info(f"Modeling: Catboost - fit done.")

        return self

    def predict(self, X):

        # Make sure that:
        # - cat features are only `str` or `int` types
        # - num features are only `float` type
        X[list(self.cat_features_set)] = X[list(self.cat_features_set)].astype(str)
        X[list(self.num_features_set)] = X[list(self.num_features_set)].astype(float)
        
        # During prediction pass features in exactly the same order as they were used to train Catboost
        prediction = self.model_calibr.predict_proba(X[self.estimator.feature_names_])
        logger.info(f"Modeling: Catboost - prediction done.")
        logger.info("=" * 50)

        return prediction
