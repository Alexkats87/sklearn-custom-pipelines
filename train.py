import pandas as pd
import numpy as np
import logging
import joblib
import json
import os

from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from transformers.featurizers import *
from transformers.models import *
from transformers.encoders import *
from transformers.const import *


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


FILE_DATA = "..."


if __name__ == "__main__":
    
 
    
    df = pd.read_csv(FILE_DATA, sep=';')
    
    y = df[TARGET]
    X = df.drop(TARGET, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    num_features_missings = {
        "battery_level": 50.0,
        'gps_location_lat': X_train['gps_location_lat'].mean(),
        "gps_location_lon": X_train['gps_location_lat'].mean(),
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
    
    model_ppl = Pipeline(steps=[
        ("simple_features_tr", SimpleFeaturesTransformer(num_features_missings, cat_features_missings) ),
        ("feature_elimination_tr", FeatureEliminationTransformer()),
        ("rare_encoder_tr", RareCategoriesTransformer()),
        ("binning_num_tr", BinningNumericalTransformer()),
        ("binning_cat_tr", BinningCategoriesTransformer()),
        ("woe_tr", WoeEncoderTransformer()),
        ("feature_elimination_tr2", FeatureEliminationTransformer(features_pattern=fr"\w+__bin__woe$")),
        ("decorr_tr", DecorrelationTransformer()),
        ("model_tr", CustomLogisticRegressionClassifier()),

    ])
    
    model_ppl.fit(X_train, y_train)
    
    y_train_pred = model_ppl.predict(X_train)
    y_test_pred = model_ppl.predict(X_test)
    
    print(f"ROC_AUC train: {roc_auc_score(y_train, y_train_pred)}")
    print(f"ROC_AUC test: {roc_auc_score(y_test, y_test_pred)}")
    
 
    
    
    
    