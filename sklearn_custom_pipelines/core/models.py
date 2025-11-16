"""Model classifiers for sklearn-custom-pipelines."""

import re
import pandas as pd
import numpy as np
import logging

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
import statsmodels.api as sm

from sklearn_custom_pipelines.utils.const import BIN, WOE, CAT, NUM

logger = logging.getLogger(__name__)


class CustomLogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression classifier with iterative feature selection.

    Fits a Logistic Regression model using statsmodels and performs
    iterative feature selection by excluding features with high p-values.

    Parameters
    ----------
    features_pattern : str, default=r"\\w+{BIN}{WOE}$"
        Regex pattern to identify features to use in the model
    p_value_max : float, default=0.05
        Maximum p-value threshold for feature retention
    """

    p_value_max = 0.05

    def __init__(self, features_pattern=fr"\w+{BIN}{WOE}$"):
        self.estimator = None
        self.features_pattern = re.compile(features_pattern)
        self.model_stats = None
        self.features_lst = None

    def fit(self, X, y):
        """Fit the model with iterative feature selection."""
        self.features_lst = list(
            filter(lambda x: self.features_pattern.match(x), X.columns)
        )

        model_stats = None

        while True:

            self.estimator = sm.Logit(
                y,
                sm.add_constant(X[self.features_lst])
            ).fit(disp=0)

            model_stats = self.estimator.summary2().tables[1]

            worst = model_stats['P>|z|'].idxmax()
            worst_p = model_stats.loc[worst]['P>|z|']

            if worst_p > self.p_value_max and worst in self.features_lst:

                model_stats = model_stats.drop(worst)
                self.features_lst.remove(worst)

                logger.debug(f"==========================")
                logger.debug(f"Features cnt: {len(self.features_lst)}")
                logger.debug(f"Dropped:      {worst}")
                logger.debug(f"P-value:      {worst_p:0.4}")

            else:
                break

        logger.info(
            f"LogisticRegression - features amount: {len(self.features_lst)}"
        )
        logger.info(f"LogisticRegression - fit done.")

        self.model_stats = model_stats
        logger.debug(model_stats)

        return self

    def predict(self, X):
        """Make predictions with the fitted model."""
        prediction = self.estimator.predict(
            sm.add_constant(X[self.features_lst], has_constant='add')
        ).tolist()

        # For test and batch prediction: returns full dataset with prediction column
        if X.shape[0] > 1:
            X = X.copy()
            X['y_pred'] = prediction
            return X

        # For prod and single prediction: compose output JSON
        features_bins_list = [f[:-len(WOE)] for f in self.features_lst]

        output = {}
        output['WOE'] = X[self.features_lst].round(3).to_dict(orient='records')[0]
        output['Bins'] = X[features_bins_list].to_dict(orient='records')[0]
        output['Score'] = round(prediction[0], 5)

        logger.info(f"LogisticRegression - prediction done.")

        return output


class CustomCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    CatBoost classifier with iterative feature selection and calibration.

    Fits a CatBoost model and performs iterative feature selection
    by excluding features with low feature importance. Includes
    probability calibration for production use.

    Parameters
    ----------
    **params
        Additional parameters to pass to CatBoostClassifier
    """

    def __init__(self, verbose=False, **params):
        self.params = params
        self.params['verbose'] = verbose
        self.estimator = CatBoostClassifier(**self.params)
        self.cat_features_set = None
        self.num_features_set = None
        self.model_calibr = None
        self.feature_names_ = None
        self.feature_importances_df = None

    def set_params(self, **params):
        """Set parameters for the CatBoost model."""
        self.params.update(params)
        self.estimator = CatBoostClassifier(**self.params)
        return self

    def get_params(self, deep=True):
        """Get parameters for the CatBoost model."""
        return self.params.copy()

    def _calc_feature_importance(self):
        """Calculate and return feature importances."""
        importance_lst = self.estimator.get_feature_importance()
        features_lst = self.estimator.feature_names_

        df_cb_importance = pd.DataFrame(
            index=features_lst,
            data=importance_lst
        ).reset_index().rename(
            columns={
                0: 'importance',
                'index': 'feature'
            }
        )

        df_cb_importance = df_cb_importance.sort_values(
            by='importance',
            ascending=False
        ).reset_index(drop=True)

        return df_cb_importance

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, **params):
        """Fit the CatBoost model with iterative feature selection."""
        self.cat_features_set = set(
            filter(lambda x: re.match(r".*__bin$", x), X.columns)
        )
        self.num_features_set = set(
            filter(lambda x: re.match(r"^(num__)", x), X.columns)
        )

        logger.info(f"CatBoost - params: {self.estimator.get_params()}")
        logger.info(
            f"CatBoost - Initial features: {len(self.cat_features_set | self.num_features_set)}"
        )

        X_valid = X.sample(frac=0.05, random_state=0)
        X_train = X.drop(X_valid.index)

        y_valid = y[X_valid.index]
        y_train = y[X_train.index]

        # Feature selection process based on CatBoost feature importances
        iteration = 1
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

            features_to_drop_set = set(
                [key for key, value in importance_dct.items() if value <= 0.01]
            )

            if not features_to_drop_set:
                logger.info(
                    f"CatBoost - Selected features: {len(self.cat_features_set | self.num_features_set)}"
                )
                break
            else:
                self.cat_features_set -= features_to_drop_set
                self.num_features_set -= features_to_drop_set
                logger.info(
                    f"Iter. {iteration}, dropped features: {len(features_to_drop_set)}"
                )

            iteration += 1

        try:
            iterations_count = self.estimator.get_best_iteration() + 1
            logger.info(f"Best iterations: {iterations_count}")
        except Exception as e:
            logger.warning(f"Can't calculate iterations: {e}")
            iterations_count = self.estimator.get_params().get('iterations', 100)

        self.params['cat_features'] = list(self.cat_features_set)
        if 'iterations' in self.params or iterations_count:
            self.params['iterations'] = iterations_count

        logger.info(f"CatBoost - final params: {self.estimator.get_params()}")

        # Final fit with selected features
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

        # Calibrate model for probability estimates
        self.model_calibr = CalibratedClassifierCV(
            estimator=CatBoostClassifier(**self.params),
            method='sigmoid'
        )

        self.model_calibr.fit(
            X=X[list(self.cat_features_set | self.num_features_set)],
            y=y
        )

        self.feature_names_ = self.estimator.feature_names_
        self.feature_importances_df = self._calc_feature_importance()

        logger.info(f"CatBoost - fit done.")

        return self

    def predict(self, X):
        """Make predictions with the fitted model."""
        X = X.copy()

        # Ensure proper data types
        X[list(self.cat_features_set)] = X[list(self.cat_features_set)].astype(str)
        X[list(self.num_features_set)] = X[list(self.num_features_set)].astype(float)

        # Use calibrated model for predictions
        prediction = self.model_calibr.predict_proba(
            X[self.estimator.feature_names_]
        )

        logger.info(f"CatBoost - prediction done.")
        logger.info("=" * 50)

        return prediction

    def predict_proba(self, X):
        """Get probability estimates for each class."""
        return self.predict(X)
