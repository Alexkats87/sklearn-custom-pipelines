"""Example training script for CatBoost model with sklearn-custom-pipelines.

This script can be run directly without installation using:
    python examples/example_catboost.py
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
    DecorrelationTransformer,
        RareCategoriesTransformer,
        PairedFeaturesTransformer,
        PairedBinaryFeaturesTransformer,
        CustomCatBoostClassifier,
)
from sklearn_custom_pipelines.utils.const import NUM, MISSING

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def create_synthetic_data(n_samples=5000, random_state=0):
    """Create synthetic dataset for demonstration."""
    np.random.seed(random_state)
    
    # Create features with actual signal
    battery_level = np.random.uniform(0, 100, n_samples)
    gps_lat = np.random.uniform(-90, 90, n_samples)
    gps_lon = np.random.uniform(-180, 180, n_samples)
    
    # Create target with some signal from features
    y = (
        (battery_level > 50).astype(int) * 0.4 +
        (gps_lat > 0).astype(int) * 0.3 +
        (gps_lon > 0).astype(int) * 0.3 +
        np.random.binomial(1, 0.1, n_samples)  # 10% random noise
    ) > 0.5
    
    data = {
        'battery_level': battery_level,
        'gps_location_lat': gps_lat,
        'gps_location_lon': gps_lon,
        'is_charging': np.random.choice(['Yes', 'No', None], n_samples, p=[0.4, 0.4, 0.2]),
        'device_model': np.random.choice(['iPhone', 'Samsung', 'Xiaomi', None], n_samples),
        'device_type': np.random.choice(['Phone', 'Tablet', None], n_samples),
        'is_vpn_connected': np.random.choice(['Yes', 'No', None], n_samples),
        'os_version': np.random.choice(['Android 10', 'Android 11', 'iOS 14', None], n_samples),
        'telco_carrier': np.random.choice(['Carrier1', 'Carrier2', 'Carrier3', None], n_samples),
        'network_type': np.random.choice(['WiFi', '4G', '5G', None], n_samples),
        # Binary flag features (as strings '0'/'1') for PairedBinaryFeaturesTransformer
        'cat__flag__a': np.random.binomial(1, 0.15, n_samples).astype(str),
        'cat__flag__b': np.random.binomial(1, 0.25, n_samples).astype(str),
        'cat__flag__c': np.random.binomial(1, 0.10, n_samples).astype(str),
        'y': y.astype(int)
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    
    # Create synthetic data
    df = create_synthetic_data(n_samples=500)
    
    y = df['y']
    X = df.drop('y', axis=1)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
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
        ("decorr_tr", DecorrelationTransformer(features_pattern=fr"^{NUM}.*")),
        ("rare_encoder_tr", RareCategoriesTransformer()),
        # Paired binary interactions (works if any binary flag features exist)
        ("paired_bin_tr", PairedBinaryFeaturesTransformer(features_pattern=r"cat__flag__")),
        ("feature_elimination_tr2", FeatureEliminationTransformer()),
        ("model_tr", CustomCatBoostClassifier(verbose=False, iterations=50)),
    ])
    
    # Fit pipeline with evaluation set
    model_ppl.fit(
        X=X_train,
        y=y_train,
        model_tr__eval_set=(
            X_train.sample(frac=0.1, random_state=0),
            y_train.sample(frac=0.1, random_state=0)
        )
    )
    
    # Make predictions
    y_train_pred = model_ppl.predict(X_train)
    y_test_pred = model_ppl.predict(X_test)
    
    print(f"\nROC_AUC train: {roc_auc_score(y_train, y_train_pred[:, 1]):.4f}")
    print(f"ROC_AUC test:  {roc_auc_score(y_test, y_test_pred[:, 1]):.4f}")
    
    print(f"\nPipeline trained successfully!")
