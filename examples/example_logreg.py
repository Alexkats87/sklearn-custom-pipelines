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
        PairedFeaturesTransformer,
        PairedBinaryFeaturesTransformer,
    CustomLogisticRegressionClassifier,
)
from sklearn_custom_pipelines.utils.const import NUM, BIN, WOE, MISSING

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def create_synthetic_data(n_samples=5000, random_state=42):
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
        # Binary flag features for PairedBinaryFeaturesTransformer (strings)
        'cat__flag__a': np.random.binomial(1, 0.15, n_samples).astype(str),
        'cat__flag__b': np.random.binomial(1, 0.25, n_samples).astype(str),
        'cat__flag__c': np.random.binomial(1, 0.10, n_samples).astype(str),
    }
    
    df = pd.DataFrame(data)

    # Additional informative numeric features
    df['usage_minutes'] = np.random.exponential(scale=30, size=n_samples)
    df['num_errors'] = np.random.poisson(1.5, n_samples)
    df['app_count'] = np.random.randint(1, 25, n_samples)
    df['device_age_months'] = np.random.randint(1, 72, n_samples)

    # Additional categorical feature with signal
    df['recent_activity'] = np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.35, 0.45, 0.2])
    df['user_segment'] = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])

    # Map categorical to numeric for signal construction
    recent_map = {'low': 0, 'medium': 1, 'high': 2}
    df['recent_activity_num'] = df['recent_activity'].map(recent_map)

    # Build a logistic signal for y using several features (plus noise)
    # Convert some categorical indicators to numeric
    is_charging_flag = (df['is_charging'] == 'Yes').astype(float)

    # linear combination -> probability via sigmoid
    logits = (
        -3.0
        + 0.02 * df['battery_level']
        + 0.03 * df['usage_minutes']
        + 0.5 * df['recent_activity_num']
        + 0.15 * df['app_count']
        - 0.2 * df['num_errors']
        + 0.6 * is_charging_flag
    )
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Sample y from Bernoulli(probs)
    df['y'] = np.random.binomial(1, probs)

    return df


if __name__ == "__main__":
    
    # Create synthetic data
    # create a larger dataset so the test split contains more objects
    df = create_synthetic_data(n_samples=5000)
    
    y = df['y']
    X = df.drop('y', axis=1)

    # Split into train and test
    # increase test_size so the test set contains more examples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
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
        # Add paired categorical interactions on binned features
        ("paired_features_tr", PairedFeaturesTransformer(features_pattern=fr".*{BIN}$")),
        # Paired binary interactions for flag features (if present)
        ("paired_bin_tr", PairedBinaryFeaturesTransformer(features_pattern=r"cat__flag__")),
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
