"""Example training script for Logistic Regression model with sklearn-custom-pipelines.

This script can be run directly without installation using:
    python examples/example_logreg.py
"""

import sys
import os
import logging

# Add the parent directory to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn_custom_pipelines import (
    SimpleFeaturesTransformer,
    FeatureEliminationTransformer,
    RareCategoriesTransformer,
    BinningNumericalTransformer,
    BinningCategoriesTransformer,
    WoeEncoderTransformer,
    CustomLogisticRegressionClassifier,
)
from sklearn_custom_pipelines.utils.const import NUM, BIN, WOE, MISSING

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def create_synthetic_data(n_samples=500, random_state=42):
    """Create synthetic dataset for demonstration."""
    np.random.seed(random_state)
    
    data = {
        'battery_level': np.random.uniform(0, 100, n_samples),
        'gps_location_lat': np.random.uniform(-90, 90, n_samples),
        'gps_location_lon': np.random.uniform(-180, 180, n_samples),
        'is_charging': np.random.choice(['Yes', 'No', None], n_samples, p=[0.4, 0.4, 0.2]),
        'device_model': np.random.choice(['iPhone', 'Samsung', 'Xiaomi', None], n_samples),
        'device_type': np.random.choice(['Phone', 'Tablet', None], n_samples),
        'is_vpn_connected': np.random.choice(['Yes', 'No', None], n_samples),
        'os_version': np.random.choice(['Android 10', 'Android 11', 'iOS 14', None], n_samples),
        'telco_carrier': np.random.choice(['Carrier1', 'Carrier2', 'Carrier3', None], n_samples),
        'network_type': np.random.choice(['WiFi', '4G', '5G', None], n_samples),
        'y': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    
    # Create synthetic data
    df = create_synthetic_data(n_samples=500)
    
    y = df['y']
    X = df.drop('y', axis=1)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define features and how to fill missings
    num_features_missings = {
        "battery_level": 50.0,
        'gps_location_lat': X_train['gps_location_lat'].mean(),
        "gps_location_lon": X_train['gps_location_lon'].mean(),
    }
    
    cat_features_missings = {
        'is_charging': MISSING,
        "device_model": MISSING,
        "device_type": MISSING,
        "is_vpn_connected": MISSING,
        "os_version": MISSING,
        "telco_carrier": MISSING,
        "network_type": MISSING,
    }
    
    # Create sklearn pipeline
    model_ppl = Pipeline(steps=[
        ("simple_features_tr", SimpleFeaturesTransformer(
            num_features_missings, cat_features_missings
        )),
        ("feature_elimination_tr", FeatureEliminationTransformer()),
        ("rare_encoder_tr", RareCategoriesTransformer()),
        ("binning_num_tr", BinningNumericalTransformer()),
        ("binning_cat_tr", BinningCategoriesTransformer()),
        ("woe_tr", WoeEncoderTransformer()),
        ("feature_elimination_tr2", FeatureEliminationTransformer(
            features_pattern=fr"\w+{BIN}{WOE}$"
        )),
        ("model_tr", CustomLogisticRegressionClassifier()),
    ])
    
    # Fit pipeline
    model_ppl.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model_ppl.predict(X_train)
    y_test_pred = model_ppl.predict(X_test)
    
    # For single row predictions, the output is a dict, so we need to handle batch predictions differently
    if isinstance(y_train_pred, pd.DataFrame):
        y_train_scores = y_train_pred['y_pred'].values
        y_test_scores = y_test_pred['y_pred'].values
    else:
        y_train_scores = y_train_pred
        y_test_scores = y_test_pred
    
    print(f"\nROC_AUC train: {roc_auc_score(y_train, y_train_scores):.4f}")
    print(f"ROC_AUC test:  {roc_auc_score(y_test, y_test_scores):.4f}")
    
    print(f"\nPipeline trained successfully!")
